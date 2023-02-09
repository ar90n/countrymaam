package countrymaam

import (
	"context"
	"encoding/gob"
	"errors"
	"io"
	"math"
	"math/rand"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/ar90n/countrymaam/pipeline"
)

const streamBufferSize = 64
const queueCapcitySize = 64

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
	Elements []treeElement[T, U]
	Roots    []treeRoot[T, U]
	config   TreeConfig
	env      linalg.Env[T]
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

func (bsp *bspTreeIndex[T, U, C]) buildRoot(ctx context.Context, id uint) (treeRoot[T, U], error) {
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
	node, err := bsp.buildTree(ctx, indice, 0, id)
	if err != nil {
		return treeRoot[T, U]{}, err
	}
	bsp.Roots[id].Nodes[0] = bsp.Roots[id].Nodes[node]

	return bsp.Roots[id], nil
}

func (bsp *bspTreeIndex[T, U, C]) buildTree(ctx context.Context, indice []int, offset uint, id uint) (uint, error) {
	nElements := uint(len(indice))
	if nElements == 0 {
		return 0, nil
	}

	nLeafs := bsp.config.Leafs
	if nLeafs == 0 {
		nLeafs = 1
	}
	if nElements <= nLeafs {
		node := treeNode[T, U]{
			Begin: offset,
			End:   offset + nElements,
		}
		nNodes := uint(len(bsp.Roots[id].Nodes))
		bsp.Roots[id].Nodes = append(bsp.Roots[id].Nodes, node)
		return nNodes, nil
	}

	cutPlane, err := (*new(C)).Construct(bsp.Elements, indice, bsp.env, bsp.config.CutPlaneOptions)
	if err != nil {
		return 0, err
	}

	mid := collection.Partition(indice, func(i int) bool {
		return cutPlane.Evaluate(bsp.Elements[i].Feature, bsp.env)
	})
	if mid == 0 || mid == uint(len(indice)) {
		mid = uint(len(indice)) / 2
	}

	left, err := bsp.buildTree(ctx, indice[:mid], offset, id)
	if err != nil {
		return 0, err
	}

	right, err := bsp.buildTree(ctx, indice[mid:], offset+mid, id)
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

	nTrees := bsp.config.Trees
	if nTrees == 0 {
		nTrees = 1
	}

	taskStream := pipeline.Seq(ctx, nTrees)
	rootStream := make(chan result[treeRoot[T, U]])
	defer close(rootStream)

	bsp.Roots = make([]treeRoot[T, U], nTrees)
	nProcs := bsp.config.Procs
	if nProcs == 0 {
		//nProcs = uint(runtime.NumCPU())
		nProcs = uint(1)
	}
	for i := uint(0); i < nProcs; i++ {
		go func() {
			for t := range taskStream {
				root, err := bsp.buildRoot(ctx, uint(t))
				rootStream <- result[treeRoot[T, U]]{Value: root, Error: err}
			}
		}()
	}

	i := 0
	for uint(i) < nTrees {
		ret := <-rootStream
		if ret.Error != nil {
			return ret.Error
		}

		bsp.Roots[i] = ret.Value
		i++
	}
	return nil
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
					distance := float64(bsp.env.SqL2(query, elem.Feature))
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
			sqDistanceToCutPlane := node.CutPlane.Distance(query, bsp.env)
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

func NewKdTreeIndex[T linalg.Number, U comparable](config TreeConfig, opts linalg.LinAlgOptions) *bspTreeIndex[T, U, kdCutPlane[T, U]] {
	gob.Register(kdCutPlane[T, U]{})

	env := linalg.NewLinAlg[T](opts)
	return &bspTreeIndex[T, U, kdCutPlane[T, U]]{
		config:   config,
		Elements: make([]treeElement[T, U], 0),
		env:      env,
	}
}

func LoadKdTreeIndex[T linalg.Number, U comparable](r io.Reader, opts linalg.LinAlgOptions) (*bspTreeIndex[T, U, kdCutPlane[T, U]], error) {
	gob.Register(kdCutPlane[T, U]{})
	index, err := loadIndex[bspTreeIndex[T, U, kdCutPlane[T, U]]](r)
	if err != nil {
		return nil, err
	}
	index.env = linalg.NewLinAlg[T](opts)

	return &index, nil
}

func NewRpTreeIndex[T linalg.Number, U comparable](config TreeConfig, opts linalg.LinAlgOptions) *bspTreeIndex[T, U, rpCutPlane[T, U]] {
	gob.Register(rpCutPlane[T, U]{})

	env := linalg.NewLinAlg[T](opts)
	return &bspTreeIndex[T, U, rpCutPlane[T, U]]{
		config:   config,
		Elements: make([]treeElement[T, U], 0),
		env:      env,
	}
}

func LoadRpTreeIndex[T linalg.Number, U comparable](r io.Reader, opts linalg.LinAlgOptions) (*bspTreeIndex[T, U, rpCutPlane[T, U]], error) {
	gob.Register(rpCutPlane[T, U]{})
	index, err := loadIndex[bspTreeIndex[T, U, rpCutPlane[T, U]]](r)
	if err != nil {
		return nil, err
	}
	index.env = linalg.NewLinAlg[T](opts)

	return &index, nil
}
