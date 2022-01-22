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
	Elements []*treeElement[T, U]
	Left     *treeNode[T, U]
	Right    *treeNode[T, U]
}

type bspTreeIndex[T number.Number, U any, C CutPlane[T, U]] struct {
	Dim           uint
	Pool          []*treeElement[T, U]
	Roots         []*treeNode[T, U]
	LeafSize      uint
	MaxCandidates uint
}

func (bsp *bspTreeIndex[T, U, C]) Add(feature []T, item U) {
	bsp.Pool = append(bsp.Pool, &treeElement[T, U]{
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

	var buildTree func(elements []*treeElement[T, U]) (*treeNode[T, U], error)
	buildTree = func(elements []*treeElement[T, U]) (*treeNode[T, U], error) {
		nElements := uint(len(elements))
		if nElements == 0 {
			return nil, nil
		}

		if nElements <= bsp.LeafSize {
			return &treeNode[T, U]{
				Elements: elements,
			}, nil
		}

		cutPlane, err := (*new(C)).Construct(elements)
		if err != nil {
			return nil, err
		}

		leftElements, rightElements := collection.Partition(elements,
			func(element *treeElement[T, U]) bool {
				return cutPlane.Evaluate(element.Feature)
			})
		left, err := buildTree(leftElements)
		if err != nil {
			return nil, err
		}
		right, err := buildTree(rightElements)
		if err != nil {
			return nil, err
		}

		return &treeNode[T, U]{
			Elements: elements,
			Left:     left,
			Right:    right,
			CutPlane: cutPlane,
		}, nil
	}

	for i := range bsp.Roots {
		elements := append([]*treeElement[T, U]{}, bsp.Pool...)
		rand.Shuffle(len(elements), func(i, j int) { elements[i], elements[j] = elements[j], elements[i] })

		root, err := buildTree(elements)
		if err != nil {
			return fmt.Errorf("build %d-th index failed", i)
		}
		bsp.Roots[i] = root
	}
	return nil
}

func (bsp *bspTreeIndex[T, U, C]) Search(query []T, n uint, r float64) ([]Candidate[U], error) {
	if !bsp.HasIndex() {
		bsp.Build()
	}

	capacity := number.Min(bsp.MaxCandidates, 5*n)
	itemQueue := collection.NewUniquePriorityQueue[*U](int(capacity))
	nodeQueue := collection.NewPriorityQueue[*treeNode[T, U]](int(capacity))
	for i, root := range bsp.Roots {
		if root == nil {
			return nil, fmt.Errorf("%d-th index is not created", i)
		}

		nodeQueue.Push(root, float64(math.MaxFloat32))
	}

	for uint(itemQueue.Len()) < bsp.MaxCandidates && 0 < nodeQueue.Len() {
		node, err := nodeQueue.Pop()
		if err != nil {
			return nil, err
		}
		if node == nil {
			continue
		}

		if item, err := itemQueue.PeekWithPriority(int(n - 1)); err == nil {
			r = item.Priority
		}

		if node.Left == nil && node.Right == nil {
			for _, element := range node.Elements {
				distance := number.CalcSqDistance(query, element.Feature)
				if distance < r {
					itemQueue.Push(&element.Item, float64(distance))
				}
			}
		} else {
			distanceToCutPlane := node.CutPlane.Distance(query)
			if -r < distanceToCutPlane {
				nodeQueue.Push(node.Right, distanceToCutPlane)
			}
			if distanceToCutPlane < r {
				nodeQueue.Push(node.Left, -distanceToCutPlane)
			}
		}
	}

	items := make([]Candidate[U], number.Min(n, uint(itemQueue.Len())))
	for i := range items {
		item, err := itemQueue.PeekWithPriority(0)
		if err != nil {
			return nil, err
		}
		itemQueue.Pop()

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
