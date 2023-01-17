package countrymaam

import (
	"context"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
)

type treeElement[T linalg.Number, U any] struct {
	Feature []T
	Item    U
}

type treeNode[T linalg.Number, U any] struct {
	CutPlane CutPlane[T, U]
	Begin    uint
	End      uint
	Left     *treeNode[T, U]
	Right    *treeNode[T, U]
}

type bspTreeIndex[T linalg.Number, U any, C CutPlane[T, U]] struct {
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
	for i := range bsp.Roots {
		bsp.Roots[i] = nil
	}
}

func (bsp *bspTreeIndex[T, U, C]) Build() error {
	if len(bsp.Pool) == 0 {
		return errors.New("empty pool")
	}

	var buildTree func(indice []int, begin uint) (*treeNode[T, U], error)
	buildTree = func(indice []int, begin uint) (*treeNode[T, U], error) {
		nElements := uint(len(indice))
		if nElements == 0 {
			return nil, nil
		}

		if nElements <= bsp.LeafSize {
			return &treeNode[T, U]{
				Begin: begin,
				End:   begin + nElements,
			}, nil
		}

		cutPlane, err := (*new(C)).Construct(bsp.Pool, indice, bsp.env)
		if err != nil {
			return nil, err
		}

		mid := collection.Partition(indice,
			func(i int) bool {
				return cutPlane.Evaluate(bsp.Pool[i].Feature, bsp.env)
			})
		left, err := buildTree(indice[:mid], begin)
		if err != nil {
			return nil, err
		}
		right, err := buildTree(indice[mid:], begin+mid)
		if err != nil {
			return nil, err
		}

		return &treeNode[T, U]{
			Begin:    begin,
			End:      begin + nElements,
			Left:     left,
			Right:    right,
			CutPlane: cutPlane,
		}, nil
	}

	bsp.Indice = make([][]int, len(bsp.Roots))
	for i := range bsp.Roots {
		indice := make([]int, len(bsp.Pool))
		for j := range indice {
			indice[j] = j
		}
		rand.Shuffle(len(indice), func(i, j int) { indice[i], indice[j] = indice[j], indice[i] })
		bsp.Indice[i] = indice
	}

	for i := range bsp.Roots {
		root, err := buildTree(bsp.Indice[i], 0)
		if err != nil {
			return fmt.Errorf("build %d-th index failed", i)
		}
		bsp.Roots[i] = root
	}
	return nil
}

type nodeQueueItem[T linalg.Number, U any] struct {
	Node      *treeNode[T, U]
	TreeIndex int
}

func (bsp *bspTreeIndex[T, U, C]) Search(query []T, n uint, maxCandidates uint) ([]Candidate[U], error) {
	if !bsp.HasIndex() {
		bsp.Build()
	}

	itemQueue := collection.NewItemQueue[*U](int(n))
	nodeQueue := collection.NewPriorityQueue[nodeQueueItem[T, U]](int(maxCandidates))
	for i, root := range bsp.Roots {
		if root == nil {
			return nil, fmt.Errorf("%d-th index is not created", i)
		}

		nodeQueue.Push(nodeQueueItem[T, U]{
			Node:      root,
			TreeIndex: i,
		}, -math.MaxFloat32)
	}

	nTotalCandidates := uint(0)
	for nTotalCandidates < maxCandidates && 0 < nodeQueue.Len() {
		nodeWithPriority, err := nodeQueue.PopWithPriority()
		if err != nil {
			return nil, err
		}

		worstPriority := nodeWithPriority.Priority
		treeIndex := nodeWithPriority.Item.TreeIndex
		node := nodeWithPriority.Item.Node
		if node == nil {
			continue
		}
		indice := bsp.Indice[treeIndex]

		if node.Left == nil && node.Right == nil {
			for i := node.Begin; i < node.End; i++ {
				distance := bsp.env.SqL2(query, bsp.Pool[indice[i]].Feature)
				itemQueue.Push(&bsp.Pool[indice[i]].Item, distance)
				nTotalCandidates++
			}
		} else {
			sqDistanceToCutPlane := node.CutPlane.Distance(query, bsp.env)
			rightPriority := linalg.Max(-float32(sqDistanceToCutPlane), worstPriority)
			nodeQueue.Push(nodeQueueItem[T, U]{Node: node.Right, TreeIndex: treeIndex}, rightPriority)

			leftPriority := linalg.Max(float32(sqDistanceToCutPlane), worstPriority)
			nodeQueue.Push(nodeQueueItem[T, U]{Node: node.Left, TreeIndex: treeIndex}, leftPriority)
		}
	}

	items := make([]Candidate[U], linalg.Min(n, uint(itemQueue.Len())))
	for i := range items {
		item, err := itemQueue.Pop()
		if err != nil {
			return nil, err
		}

		items[i].Item = *item.Item
		items[i].Distance = float64(item.Priority)
	}

	return items, nil
}

func (bsp *bspTreeIndex[T, U, C]) Search2(ctx context.Context, query []T) <-chan Candidate[U] {
	ch := make(chan Candidate[U])

	go func() {
		defer close(ch)

		if !bsp.HasIndex() {
			bsp.Build()
		}

		maxCandidates := 64
		nodeQueue := collection.NewPriorityQueue[nodeQueueItem[T, U]](int(maxCandidates))
		for i, root := range bsp.Roots {
			if root == nil {
				//return nil, fmt.Errorf("%d-th index is not created", i)
				return
			}

			nodeQueue.Push(nodeQueueItem[T, U]{
				Node:      root,
				TreeIndex: i,
			}, -math.MaxFloat32)
		}

		ch2 := make(chan Candidate[U])
		go func() {
			defer close(ch2)
			//for nodeWithPriority := range nodeQueue.PopWithPriority2() {
			for 0 < nodeQueue.Len() {
				nodeWithPriority, err := nodeQueue.PopWithPriority()
				if err != nil {
					return
				}

				worstPriority := nodeWithPriority.Priority
				treeIndex := nodeWithPriority.Item.TreeIndex
				node := nodeWithPriority.Item.Node
				if node == nil {
					continue
				}
				indice := bsp.Indice[treeIndex]

				if node.Left == nil && node.Right == nil {
					for i := node.Begin; i < node.End; i++ {
						distance := bsp.env.SqL2(query, bsp.Pool[indice[i]].Feature)
						ch2 <- Candidate[U]{
							Item:     bsp.Pool[indice[i]].Item,
							Distance: float64(distance),
						}
					}
				} else {
					sqDistanceToCutPlane := node.CutPlane.Distance(query, bsp.env)
					rightPriority := linalg.Max(-float32(sqDistanceToCutPlane), worstPriority)
					nodeQueue.Push(nodeQueueItem[T, U]{Node: node.Right, TreeIndex: treeIndex}, rightPriority)

					leftPriority := linalg.Max(float32(sqDistanceToCutPlane), worstPriority)
					nodeQueue.Push(nodeQueueItem[T, U]{Node: node.Left, TreeIndex: treeIndex}, leftPriority)
				}
			}
		}()

		for candidate := range ch2 {
			select {
			case <-ctx.Done():
				return
			case ch <- candidate:
			}
		}
		return
	}()

	return ch
}

func (bsp bspTreeIndex[T, U, C]) HasIndex() bool {
	for i := range bsp.Roots {
		if bsp.Roots[i] == nil {
			return false
		}
	}

	return true
}

func (bsp bspTreeIndex[T, U, C]) Save(w io.Writer) error {
	return saveIndex(&bsp, w)
}
