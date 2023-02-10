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

type treeRoot[T linalg.Number, U comparable, C CutPlane[T, U]] struct {
	Indice []int
	Nodes  []treeNode[T, U]
}

func (r *treeRoot[T, U, C]) addNode(node treeNode[T, U]) uint {
	nc := uint(len(r.Nodes))
	r.Nodes = append(r.Nodes, node)

	return nc
}

func (r *treeRoot[T, U, C]) Build(elements []treeElement[T, U], indice []int, leafs, offset uint, env linalg.Env[T], cutPlaneConfig CutPlaneOptions) (uint, error) {
	ec := uint(len(indice))
	if ec == 0 {
		return 0, nil
	}

	curIdx := r.addNode(treeNode[T, U]{
		Begin: offset,
		End:   offset + ec,
	})

	if ec <= leafs {
		return curIdx, nil
	}

	cutPlane, err := (*new(C)).Construct(elements, indice, env, cutPlaneConfig)
	if err != nil {
		return 0, err
	}
	r.Nodes[curIdx].CutPlane = cutPlane

	mid := collection.Partition(indice, func(i int) bool {
		return cutPlane.Evaluate(elements[i].Feature, env)
	})

	left, err := r.Build(elements, indice[:mid], leafs, offset, env, cutPlaneConfig)
	if err != nil {
		return 0, err
	}
	r.Nodes[curIdx].Left = left

	right, err := r.Build(elements, indice[mid:], leafs, offset+mid, env, cutPlaneConfig)
	if err != nil {
		return 0, err
	}
	r.Nodes[curIdx].Right = right

	return curIdx, nil
}

func newTreeRoot[T linalg.Number, U comparable, C CutPlane[T, U]](elements []treeElement[T, U], leafs uint, env linalg.Env[T], cutPlaneConfig CutPlaneOptions) (treeRoot[T, U, C], error) {
	indice := make([]int, len(elements))
	for i := range indice {
		indice[i] = i
	}
	rand.Shuffle(len(indice), func(i, j int) { indice[i], indice[j] = indice[j], indice[i] })

	root := treeRoot[T, U, C]{
		Indice: indice,
		Nodes:  []treeNode[T, U]{},
	}
	_, err := root.Build(elements, indice, leafs, 0, env, cutPlaneConfig)
	if err != nil {
		return root, err
	}

	return root, nil
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
	Roots          []treeRoot[T, U, C]
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

func (bsp *bspTreeIndex[T, U, C]) Build(ctx context.Context) error {
	if bsp.Empty() {
		return errors.New("empty pool")
	}

	env := linalg.NewLinAlgFromContext[T](ctx)

	bsp.Roots = make([]treeRoot[T, U, C], bsp.Trees)
	p := pool.New().WithMaxGoroutines(int(bsp.Procs)).WithErrors()
	for i := uint(0); i < bsp.Trees; i++ {
		i := i
		p.Go(func() error {
			root, err := newTreeRoot[T, U, C](bsp.Elements, bsp.Leafs, env, bsp.CutPlaneConfig)
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
			nodeWithPriority, err := queue.PopWithPriority()
			if err != nil {
				continue
			}

			ri := nodeWithPriority.Item.RootIdx
			root := bsp.Roots[ri]
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
