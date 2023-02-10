package countrymaam

import (
	"context"
	"encoding/gob"
	"errors"
	"io"
	"log"
	"math"
	"math/rand"
	"runtime"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/ar90n/countrymaam/pipeline"
	"github.com/sourcegraph/conc/pool"
)

const streamBufferSize = 64
const queueCapcitySize = 64
const defaultLeafs = 16

type treeElement[T linalg.Number, U comparable] struct {
	Feature []T
	Item    U
}

type treeNode[T linalg.Number, U comparable] struct {
	CutPlane CutPlane[T, U]
	Begin    uint
	End      uint
	Left     uint
	Right    uint
}

type treeRoot[T linalg.Number, U comparable] struct {
	Indice []int
	Nodes  []treeNode[T, U]
}

type TreeConfig struct {
	CutPlaneOptions
	Dim   uint
	Leafs uint
	Trees uint
	Procs uint
}

type bspTreeIndex[T linalg.Number, U comparable, C CutPlane[T, U]] struct {
	Elements       []treeElement[T, U]
	Roots          []treeRoot[T, U]
	CutPlaneConfig CutPlaneOptions
	Dim            uint
	Leafs          uint
	Trees          uint
	Procs          uint
}

type queueItem struct {
	RootIdx uint
	NodeIdx uint
}

type result[T any] struct {
	Value T
	Error error
}

var _ = (*bspTreeIndex[float32, int, kdCutPlane[float32, int]])(nil)

func (bsp *bspTreeIndex[T, U, C]) Add(feature []T, item U) {
	bsp.Elements = append(bsp.Elements, treeElement[T, U]{
		Item:    item,
		Feature: feature,
	})
	bsp.ClearIndex()
}

func (bsp *bspTreeIndex[T, U, C]) buildRoot(ctx context.Context, id uint, env linalg.Env[T]) (treeRoot[T, U], error) {
	indice := pipeline.ToSlice(ctx, pipeline.Seq(ctx, uint(len(bsp.Elements))))
	rand.Shuffle(len(indice), func(i, j int) { indice[i], indice[j] = indice[j], indice[i] })

	bsp.Roots[id] = treeRoot[T, U]{
		Indice: indice,
		Nodes: []treeNode[T, U]{
			{
				Begin: 0,
				End:   uint(len(indice)),
			},
		},
	}
	node, err := bsp.buildTree(ctx, indice, 0, id, env)
	if err != nil {
		return treeRoot[T, U]{}, err
	}
	bsp.Roots[id].Nodes[0] = bsp.Roots[id].Nodes[node]

	return bsp.Roots[id], nil
}

func (bsp *bspTreeIndex[T, U, C]) buildTree(ctx context.Context, indice []int, offset uint, id uint, env linalg.Env[T]) (uint, error) {
	nElements := uint(len(indice))
	if nElements == 0 {
		return 0, nil
	}

	if nElements <= bsp.Leafs {
		node := treeNode[T, U]{
			Begin: offset,
			End:   offset + nElements,
		}
		nNodes := uint(len(bsp.Roots[id].Nodes))
		bsp.Roots[id].Nodes = append(bsp.Roots[id].Nodes, node)
		return nNodes, nil
	}

	cutPlane, err := (*new(C)).Construct(bsp.Elements, indice, env, bsp.CutPlaneConfig)
	if err != nil {
		return 0, err
	}

	mid := collection.Partition(indice, func(i int) bool {
		return cutPlane.Evaluate(bsp.Elements[i].Feature, env)
	})
	if mid == 0 || mid == uint(len(indice)) {
		mid = uint(len(indice)) / 2
	}

	left, err := bsp.buildTree(ctx, indice[:mid], offset, id, env)
	if err != nil {
		return 0, err
	}

	right, err := bsp.buildTree(ctx, indice[mid:], offset+mid, id, env)
	if err != nil {
		return 0, err
	}

	node := treeNode[T, U]{
		Begin:    offset,
		End:      offset + nElements,
		Left:     left,
		Right:    right,
		CutPlane: cutPlane,
	}
	nNodes := uint(len(bsp.Roots[id].Nodes))
	bsp.Roots[id].Nodes = append(bsp.Roots[id].Nodes, node)
	return nNodes, nil
}

func (bsp *bspTreeIndex[T, U, C]) Build(ctx context.Context) error {
	if bsp.Empty() {
		return errors.New("empty pool")
	}

	env := linalg.NewLinAlgFromContext[T](ctx)

	bsp.Roots = make([]treeRoot[T, U], bsp.Trees)
	p := pool.New().WithMaxGoroutines(int(bsp.Procs)).WithErrors()
	for i := uint(0); i < bsp.Trees; i++ {
		i := i
		p.Go(func() error {
			root, err := bsp.buildRoot(ctx, i, env)
			if err != nil {
				return err
			}
			bsp.Roots[i] = root
			return nil
		})
	}

	return p.Wait()
}

func (bsp *bspTreeIndex[T, U, C]) Search(ctx context.Context, query []T, n uint, maxCandidates uint) ([]Candidate[U], error) {
	ch := bsp.SearchChannel(ctx, query)

	items := make([]collection.WithPriority[U], 0, maxCandidates)
	for item := range ch {
		if maxCandidates <= uint(len(items)) {
			break
		}
		items = append(items, collection.WithPriority[U]{Item: item.Item, Priority: item.Distance})
	}
	pq := collection.NewPriorityQueueFromSlice(items)

	// take unique neighbors
	ret := make([]Candidate[U], 0, n)
	founds := make(map[U]struct{}, maxCandidates)
	for uint(len(ret)) < n {
		item, err := pq.PopWithPriority()
		if err != nil {
			return nil, err
		}

		if _, ok := founds[item.Item]; ok {
			continue
		}
		founds[item.Item] = struct{}{}

		ret = append(ret, Candidate[U]{Item: item.Item, Distance: item.Priority})
	}
	return ret, nil
}

func (bsp *bspTreeIndex[T, U, C]) SearchChannel(ctx context.Context, query []T) <-chan Candidate[U] {
	outputStream := make(chan Candidate[U], streamBufferSize)
	go func() error {
		defer close(outputStream)

		if !bsp.HasIndex() {
			bsp.Build(ctx)
		}

		env := linalg.NewLinAlgFromContext[T](ctx)

		queue := collection.NewPriorityQueue[queueItem](queueCapcitySize)
		for i := range bsp.Roots {
			queue.Push(queueItem{RootIdx: uint(i), NodeIdx: 0}, -math.MaxFloat64)
		}

		for 0 < queue.Len() {
			//log.Print("queue.Len()", queue.Len())
			nodeWithPriority, err := queue.PopWithPriority()
			if err != nil {
				continue
			}

			ri := nodeWithPriority.Item.RootIdx
			root := bsp.Roots[ri]
			//log.Println("root", ri, "node", nodeWithPriority.Item.NodeIdx, "priority", nodeWithPriority.Priority, "len(queue)", queue.Len(), "len(outputStream)", len(outputStream), "len(root.Nodes)", len(root.Nodes))
			node := root.Nodes[nodeWithPriority.Item.NodeIdx]
			if node.Left == 0 && node.Right == 0 {
				for i := node.Begin; i < node.End; i++ {
					elem := bsp.Elements[root.Indice[i]]
					distance := float64(env.SqL2(query, elem.Feature))
					select {
					case <-ctx.Done():
						return nil
					case outputStream <- Candidate[U]{
						Item:     elem.Item,
						Distance: distance,
					}:
					}
				}
				continue
			}

			worstPriority := nodeWithPriority.Priority
			sqDistanceToCutPlane := node.CutPlane.Distance(query, env)
			if 0 < node.Right {
				rightPriority := linalg.Max(-sqDistanceToCutPlane, worstPriority)
				queue.Push(queueItem{RootIdx: ri, NodeIdx: node.Right}, rightPriority)
			}

			if 0 < node.Left {
				leftPriority := linalg.Max(sqDistanceToCutPlane, worstPriority)
				queue.Push(queueItem{RootIdx: ri, NodeIdx: node.Left}, leftPriority)
			}
		}

		return nil
	}()

	return outputStream
}

func (bsp bspTreeIndex[T, U, C]) HasIndex() bool {
	return 0 < len(bsp.Roots)
}

func (bsp *bspTreeIndex[T, U, C]) ClearIndex() {
	bsp.Roots = nil
}

func (bsp bspTreeIndex[T, U, C]) Empty() bool {
	return len(bsp.Elements) == 0
}

func (bsp bspTreeIndex[T, U, C]) Save(w io.Writer) error {
	return saveIndex(bsp, w)
}

func NewKdTreeIndex[T linalg.Number, U comparable](config TreeConfig) (*bspTreeIndex[T, U, kdCutPlane[T, U]], error) {
	return NewTreeIndex[T, U, kdCutPlane[T, U]](config)
}

func LoadKdTreeIndex[T linalg.Number, U comparable](r io.Reader) (*bspTreeIndex[T, U, kdCutPlane[T, U]], error) {
	return LoadTreeIndex[T, U, kdCutPlane[T, U]](r)
}

func NewRpTreeIndex[T linalg.Number, U comparable](config TreeConfig) (*bspTreeIndex[T, U, rpCutPlane[T, U]], error) {
	return NewTreeIndex[T, U, rpCutPlane[T, U]](config)
}

func LoadRpTreeIndex[T linalg.Number, U comparable](r io.Reader) (*bspTreeIndex[T, U, rpCutPlane[T, U]], error) {
	return LoadTreeIndex[T, U, rpCutPlane[T, U]](r)
}

func NewTreeIndex[T linalg.Number, U comparable, C CutPlane[T, U]](config TreeConfig) (*bspTreeIndex[T, U, C], error) {
	gob.Register(*new(C))

	if config.Dim == uint(0) {
		return nil, errors.New("dimension is not set")
	}

	leafs := config.Leafs
	if leafs == 0 {
		leafs = defaultLeafs
		log.Println("Leafs in given Config is not set. use default value", leafs)
	}

	trees := config.Trees
	if trees == 0 {
		trees = 1
		log.Println("Trees in given Config is not set. use default value", trees)
	}

	procs := config.Procs
	if procs == 0 {
		procs = uint(runtime.NumCPU())
		log.Println("Procs in given Config is not set. use default value", procs)
	}

	return &bspTreeIndex[T, U, C]{
		Elements:       make([]treeElement[T, U], 0),
		CutPlaneConfig: config.CutPlaneOptions,
		Dim:            config.Dim,
		Leafs:          leafs,
		Trees:          trees,
		Procs:          procs,
	}, nil
}

func LoadTreeIndex[T linalg.Number, U comparable, C CutPlane[T, U]](r io.Reader) (*bspTreeIndex[T, U, C], error) {
	gob.Register(*new(C))

	index, err := loadIndex[bspTreeIndex[T, U, C]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}
