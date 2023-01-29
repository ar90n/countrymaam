package countrymaam

import (
	"context"
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sort"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/ar90n/countrymaam/pipeline"
)

type treeElement[T linalg.Number, U comparable] struct {
	Feature []T
	Item    U
}

type treeNode[T linalg.Number, U comparable] struct {
	CutPlane CutPlane[T, U]
	Begin    uint
	End      uint
	Left     *treeNode[T, U]
	Right    *treeNode[T, U]
}

type treeRoot[T linalg.Number, U comparable] struct {
	Indice []int
	Node   *treeNode[T, U]
}

type bspTreeIndex[T linalg.Number, U comparable, C CutPlane[T, U]] struct {
	Dim      uint
	Pool     []treeElement[T, U]
	Roots    []treeRoot[T, U]
	LeafSize uint
	env      linalg.Env[T]
	nTrees   uint
}

type mergeItem[T any] struct {
	Item T
	From int
}

var _ = (*bspTreeIndex[float32, int, kdCutPlane[float32, int]])(nil)

func (bsp *bspTreeIndex[T, U, C]) Add(feature []T, item U) {
	bsp.Pool = append(bsp.Pool, treeElement[T, U]{
		Item:    item,
		Feature: feature,
	})
	bsp.ClearIndex()
}

func (bsp *bspTreeIndex[T, U, C]) buildRoot(ctx context.Context, indice []int, offset uint) <-chan treeRoot[T, U] {
	outputStream := make(chan treeRoot[T, U])
	go func() {
		defer close(outputStream)
		node, ok := <-bsp.buildTree(ctx, indice, offset)
		if !ok {
			return
		}
		outputStream <- treeRoot[T, U]{
			Indice: indice,
			Node:   node,
		}
	}()
	return outputStream
}

func (bsp *bspTreeIndex[T, U, C]) buildTree(ctx context.Context, indice []int, offset uint) <-chan *treeNode[T, U] {
	outputStream := make(chan *treeNode[T, U])
	go func() {
		defer close(outputStream)

		nElements := uint(len(indice))
		if nElements == 0 {
			return
		}

		if nElements <= bsp.LeafSize {
			outputStream <- &treeNode[T, U]{
				Begin: offset,
				End:   offset + nElements,
			}
			return
		}

		cutPlane, err := (*new(C)).Construct(bsp.Pool, indice, bsp.env)
		if err != nil {
			return
		}

		mid := collection.Partition(indice, func(i int) bool {
			return cutPlane.Evaluate(bsp.Pool[i].Feature, bsp.env)
		})
		if mid == 0 || mid == uint(len(indice)) {
			mid = uint(len(indice)) / 2
		}

		leftStream := bsp.buildTree(ctx, indice[:mid], offset)
		rightStream := bsp.buildTree(ctx, indice[mid:], offset+mid)

		left, ok := <-leftStream
		if !ok {
			return
		}

		right, ok := <-rightStream
		if !ok {
			return
		}

		outputStream <- &treeNode[T, U]{
			Begin:    offset,
			End:      offset + nElements,
			Left:     left,
			Right:    right,
			CutPlane: cutPlane,
		}
	}()
	return outputStream
}

func (bsp *bspTreeIndex[T, U, C]) Build(ctx context.Context) error {
	if bsp.Empty() {
		return errors.New("empty pool")
	}

	rootStreams := make([]<-chan treeRoot[T, U], bsp.nTrees)
	for i := range rootStreams {
		indice := pipeline.ToSlice(ctx, pipeline.Seq(ctx, uint(len(bsp.Pool))))
		rand.Shuffle(len(indice), func(i, j int) { indice[i], indice[j] = indice[j], indice[i] })
		rootStreams[i] = bsp.buildRoot(ctx, indice, 0)
	}

	for i, ch := range rootStreams {
		root, ok := <-ch
		if !ok {
			return fmt.Errorf("build %d-th index failed", i)
		}

		bsp.Roots = append(bsp.Roots, root)
	}
	return nil
}

func (bsp *bspTreeIndex[T, U, C]) Search(ctx context.Context, query []T, n uint, maxCandidates uint) ([]Candidate[U], error) {
	ch := bsp.SearchChannel(ctx, query)
	ch = pipeline.Unique(ctx, ch)
	ch = pipeline.Take(ctx, maxCandidates, ch)
	items := pipeline.ToSlice(ctx, ch)
	sort.Slice(items, func(i, j int) bool {
		return items[i].Distance < items[j].Distance
	})

	if uint(len(items)) < n {
		n = uint(len(items))
	}
	return items[:n], nil
}

func (bsp *bspTreeIndex[T, U, C]) SearchChannel(ctx context.Context, query []T) <-chan Candidate[U] {
	outputStream := make(chan Candidate[U])
	go func() error {
		defer close(outputStream)

		if !bsp.HasIndex() {
			bsp.Build(ctx)
		}

		if !bsp.HasIndex() {
			return errors.New("no index")
		}

		nodeStreams := make([]<-chan collection.WithPriority[*treeNode[T, U]], 0, len(bsp.Roots))
		for i, root := range bsp.Roots {
			if root.Node == nil {
				return fmt.Errorf("%d-th index is not created", i)
			}

			nodeStreams = append(nodeStreams, searchNode(ctx, root.Node, query, bsp.env))
		}
		mergedStream := merge(ctx, nodeStreams...)

		for mi := range pipeline.OrDone(ctx, mergedStream) {
			node := mi.Item
			indice := bsp.Roots[mi.From].Indice
			for i := node.Begin; i < node.End; i++ {
				distance := bsp.env.SqL2(query, bsp.Pool[indice[i]].Feature)
				outputStream <- Candidate[U]{
					Item:     bsp.Pool[indice[i]].Item,
					Distance: float64(distance),
				}
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
	return len(bsp.Pool) == 0
}

func (bsp bspTreeIndex[T, U, C]) Save(w io.Writer) error {
	return saveIndex(&bsp, w)
}

func searchNode[T linalg.Number, U comparable](ctx context.Context, root *treeNode[T, U], query []T, env linalg.Env[T]) <-chan collection.WithPriority[*treeNode[T, U]] {
	outputStream := make(chan collection.WithPriority[*treeNode[T, U]])
	go func() {
		defer close(outputStream)

		maxCandidates := 64
		nodeQueue := collection.NewPriorityQueue[*treeNode[T, U]](maxCandidates)

		nodeQueue.Push(root, -math.MaxFloat32)
		for 0 < nodeQueue.Len() {
			nodeWithPriority, err := nodeQueue.PopWithPriority()
			if err != nil {
				return
			}

			node := nodeWithPriority.Item
			if node == nil {
				continue
			}

			if node.Left == nil && node.Right == nil {
				select {
				case <-ctx.Done():
					return
				case outputStream <- nodeWithPriority:
				}
			} else {
				worstPriority := nodeWithPriority.Priority
				sqDistanceToCutPlane := node.CutPlane.Distance(query, env)
				rightPriority := linalg.Max(-float32(sqDistanceToCutPlane), worstPriority)
				nodeQueue.Push(node.Right, rightPriority)

				leftPriority := linalg.Max(float32(sqDistanceToCutPlane), worstPriority)
				nodeQueue.Push(node.Left, leftPriority)
			}
		}
	}()

	return outputStream
}

func NewKdTreeIndex[T linalg.Number, U comparable](dim uint, leafSize uint, opts linalg.LinAlgOptions) *bspTreeIndex[T, U, kdCutPlane[T, U]] {
	gob.Register(kdCutPlane[T, U]{})

	env := linalg.NewLinAlg[T](opts)
	return &bspTreeIndex[T, U, kdCutPlane[T, U]]{
		Dim:      dim,
		Pool:     make([]treeElement[T, U], 0, 4096),
		LeafSize: leafSize,
		env:      env,
		nTrees:   1,
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

func NewRpTreeIndex[T linalg.Number, U comparable](dim uint, leafSize uint, opts linalg.LinAlgOptions) *bspTreeIndex[T, U, rpCutPlane[T, U]] {
	gob.Register(rpCutPlane[T, U]{})

	env := linalg.NewLinAlg[T](opts)
	return &bspTreeIndex[T, U, rpCutPlane[T, U]]{
		Dim:      dim,
		Pool:     make([]treeElement[T, U], 0),
		LeafSize: leafSize,
		env:      env,
		nTrees:   1,
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

func NewRandomizedKdTreeIndex[T linalg.Number, U comparable](dim uint, leafSize uint, nTrees uint, opts linalg.LinAlgOptions) *bspTreeIndex[T, U, randomizedKdCutPlane[T, U]] {
	gob.Register(kdCutPlane[T, U]{})
	gob.Register(randomizedKdCutPlane[T, U]{})

	env := linalg.NewLinAlg[T](opts)
	return &bspTreeIndex[T, U, randomizedKdCutPlane[T, U]]{
		Dim:      dim,
		Pool:     make([]treeElement[T, U], 0),
		LeafSize: leafSize,
		env:      env,
		nTrees:   nTrees,
	}
}

func LoadRandomizedKdTreeIndex[T linalg.Number, U comparable](r io.Reader, opts linalg.LinAlgOptions) (*bspTreeIndex[T, U, randomizedKdCutPlane[T, U]], error) {
	gob.Register(kdCutPlane[T, U]{})
	gob.Register(randomizedKdCutPlane[T, U]{})
	index, err := loadIndex[bspTreeIndex[T, U, randomizedKdCutPlane[T, U]]](r)
	if err != nil {
		return nil, err
	}
	index.env = linalg.NewLinAlg[T](opts)

	return &index, nil
}

func NewRandomizedRpTreeIndex[T linalg.Number, U comparable](dim uint, leafSize uint, nTrees uint, opts linalg.LinAlgOptions) *bspTreeIndex[T, U, rpCutPlane[T, U]] {
	gob.Register(rpCutPlane[T, U]{})

	env := linalg.NewLinAlg[T](opts)
	return &bspTreeIndex[T, U, rpCutPlane[T, U]]{
		Dim:      dim,
		Pool:     make([]treeElement[T, U], 0, 4096),
		LeafSize: leafSize,
		env:      env,
		nTrees:   nTrees,
	}
}

func LoadRandomizedRpTreeIndex[T linalg.Number, U comparable](r io.Reader, opts linalg.LinAlgOptions) (*bspTreeIndex[T, U, rpCutPlane[T, U]], error) {
	gob.Register(rpCutPlane[T, U]{})
	index, err := loadIndex[bspTreeIndex[T, U, rpCutPlane[T, U]]](r)
	if err != nil {
		return nil, err
	}
	index.env = linalg.NewLinAlg[T](opts)

	return &index, nil
}

func merge[T any](ctx context.Context, inputStreams ...<-chan collection.WithPriority[T]) <-chan mergeItem[T] {
	outputStream := make(chan mergeItem[T])
	go func() {
		defer close(outputStream)

		nodeQueue := collection.NewPriorityQueue[mergeItem[T]](len(inputStreams))
		for i, nch := range inputStreams {
			n, ok := <-nch
			if ok {
				nodeQueue.Push(mergeItem[T]{Item: n.Item, From: i}, n.Priority)
			}
		}

		for 0 < nodeQueue.Len() {
			item, err := nodeQueue.Pop()
			if err != nil {
				return
			}

			select {
			case <-ctx.Done():
				return
			case outputStream <- item:
				n, ok := <-inputStreams[item.From]
				if ok {
					nodeQueue.Push(mergeItem[T]{Item: n.Item, From: item.From}, n.Priority)
				}
			}
		}

	}()

	return outputStream
}
