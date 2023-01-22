package countrymaam

import (
	"context"
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

type bspTreeIndex[T linalg.Number, U comparable, C CutPlane[T, U]] struct {
	Dim      uint
	Pool     []treeElement[T, U]
	Indice   [][]int
	Roots    []*treeNode[T, U]
	LeafSize uint
	env      linalg.Env[T]
}

var _ = (*bspTreeIndex[float32, int, kdCutPlane[float32, int]])(nil)

func (bsp *bspTreeIndex[T, U, C]) Add(feature []T, item U) {
	bsp.Pool = append(bsp.Pool, treeElement[T, U]{
		Item:    item,
		Feature: feature,
	})
	bsp.ClearIndex()
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

	chs := make([]<-chan *treeNode[T, U], len(bsp.Roots))
	bsp.Indice = make([][]int, len(bsp.Roots))
	for i := range bsp.Roots {
		indice := pipeline.ToSlice(ctx, pipeline.Seq(ctx, uint(len(bsp.Pool))))
		rand.Shuffle(len(indice), func(i, j int) { indice[i], indice[j] = indice[j], indice[i] })
		bsp.Indice[i] = indice

		chs[i] = bsp.buildTree(ctx, indice, 0)
	}

	for i, ch := range chs {
		root, ok := <-ch
		if !ok {
			return fmt.Errorf("build %d-th index failed", i)
		}

		bsp.Roots[i] = root
	}
	return nil
}

type nodeQueueItem[T linalg.Number, U comparable] struct {
	Node      *treeNode[T, U]
	TreeIndex int
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

		nodeStreams := make([]<-chan collection.WithPriority[*treeNode[T, U]], 0, len(bsp.Roots))
		for i, root := range bsp.Roots {
			if root == nil {
				return fmt.Errorf("%d-th index is not created", i)
			}

			nodeStreams = append(nodeStreams, searchNode(ctx, root, query, bsp.env))
		}
		mergedStream := merge(ctx, nodeStreams...)

		for mi := range pipeline.OrDone(ctx, mergedStream) {
			node := mi.Item
			indice := bsp.Indice[mi.From]
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
	for i := range bsp.Roots {
		if bsp.Roots[i] == nil {
			return false
		}
	}

	return true
}

func (bsp bspTreeIndex[T, U, C]) ClearIndex() {
	for i := range bsp.Roots {
		bsp.Roots[i] = nil
	}
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

type mergeItem[T any] struct {
	Item T
	From int
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
