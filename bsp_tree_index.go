package countrymaam

import (
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/number"
)

type treeElement[T number.Number, U any] struct {
	Feature []T
	Item    U
}

type treeNode[T number.Number, U any] struct {
	CutPlane CutPlane[T, U]
	Begin    uint
	End      uint
	Left     *treeNode[T, U]
	Right    *treeNode[T, U]
}

type bspTreeIndex[T number.Number, U any, C CutPlane[T, U]] struct {
	Dim           uint
	Pool          []treeElement[T, U]
	Indice        [][]int
	Roots         []*treeNode[T, U]
	LeafSize      uint
	MaxCandidates uint
}

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

		cutPlane, err := (*new(C)).Construct(bsp.Pool, indice)
		if err != nil {
			return nil, err
		}

		mid := collection.Partition(indice,
			func(i int) bool {
				return cutPlane.Evaluate(bsp.Pool[i].Feature)
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

type nodeQueueItem[T number.Number, U any] struct {
	Node      *treeNode[T, U]
	TreeIndex int
}

func (bsp *bspTreeIndex[T, U, C]) Search(query []T, n uint, r float64) ([]Candidate[U], error) {
	if !bsp.HasIndex() {
		bsp.Build()
	}

	itemQueue := collection.NewItemQueue[*U](int(n))
	nodeQueue := collection.NewPriorityQueue[nodeQueueItem[T, U]](int(bsp.MaxCandidates))
	for i, root := range bsp.Roots {
		if root == nil {
			return nil, fmt.Errorf("%d-th index is not created", i)
		}

		nodeQueue.Push(nodeQueueItem[T, U]{
			Node:      root,
			TreeIndex: i,
		}, -float64(math.MaxFloat32))
	}

	nTotalCandidates := uint(0)
	for nTotalCandidates < bsp.MaxCandidates && 0 < nodeQueue.Len() {
		nodeWithTreeIndex, err := nodeQueue.Pop()
		if err != nil {
			return nil, err
		}
		node := nodeWithTreeIndex.Node
		if node == nil {
			continue
		}
		treeIndex := nodeWithTreeIndex.TreeIndex
		indice := bsp.Indice[treeIndex]

		if worstPriority := itemQueue.WorstPriority(); !math.IsInf(worstPriority, 1) {
			r = worstPriority
		}

		if node.Left == nil && node.Right == nil {
			for i := node.Begin; i < node.End; i++ {
				distance := number.CalcSqDistance(query, bsp.Pool[indice[i]].Feature)
				if distance < r {
					itemQueue.Push(&bsp.Pool[indice[i]].Item, float64(distance))
					nTotalCandidates++
				}
			}
		} else {
			distanceToCutPlane := node.CutPlane.Distance(query)
			if -r < distanceToCutPlane {
				nodeQueue.Push(nodeQueueItem[T, U]{Node: node.Right, TreeIndex: treeIndex}, -distanceToCutPlane)
			}
			if distanceToCutPlane < r {
				nodeQueue.Push(nodeQueueItem[T, U]{Node: node.Left, TreeIndex: treeIndex}, distanceToCutPlane)
			}
		}
	}

	items := make([]Candidate[U], number.Min(n, uint(itemQueue.Len())))
	for i := range items {
		item, err := itemQueue.Pop()
		if err != nil {
			return nil, err
		}

		items[i].Item = *item.Item
		items[i].Distance = item.Priority
	}

	return items, nil
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
