package index

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/metric"
	"github.com/ar90n/countrymaam/number"
)

type treeElement[T number.Number, U any] struct {
	Feature []T
	Item    U
}

type treeNode[T number.Number, U any] struct {
	CutPlane CutPlane[T]
	Elements []*treeElement[T, U]
	Left     *treeNode[T, U]
	Right    *treeNode[T, U]
}

type bspTreeIndex[T number.Number, U any, M metric.Metric[T]] struct {
	dim         uint
	pool        []*treeElement[T, U]
	metric      M
	roots       []*treeNode[T, U]
	leafSize    uint
	newCutPlane func(elements []*treeElement[T, U], selector func(element *treeElement[T, U]) []T) (CutPlane[T], error)
}

func NewKdTreeIndex[T number.Number, U any, M metric.Metric[T]](dim uint, leafSize uint) *bspTreeIndex[T, U, M] {
	return &bspTreeIndex[T, U, M]{
		dim:         dim,
		pool:        make([]*treeElement[T, U], 0),
		roots:       make([]*treeNode[T, U], 1),
		leafSize:    leafSize,
		newCutPlane: NewKdCutPlane[T, *treeElement[T, U]],
	}
}

func NewRpTreeIndex[T number.Number, U any, M metric.Metric[T]](dim uint, leafSize uint) *bspTreeIndex[T, U, M] {
	return &bspTreeIndex[T, U, M]{
		dim:         dim,
		pool:        make([]*treeElement[T, U], 0),
		roots:       make([]*treeNode[T, U], 1),
		leafSize:    leafSize,
		newCutPlane: NewRpCutPlane[T, *treeElement[T, U]],
	}
}

func NewRandomizedKdTreeIndex[T number.Number, U any, M metric.Metric[T]](dim uint, leafSize uint, nTrees uint) *bspTreeIndex[T, U, M] {
	return &bspTreeIndex[T, U, M]{
		dim:         dim,
		pool:        make([]*treeElement[T, U], 0),
		roots:       make([]*treeNode[T, U], nTrees),
		leafSize:    leafSize,
		newCutPlane: NewRandomizedKdCutPlane[T, *treeElement[T, U]],
	}
}

func NewRandomizedRpTreeIndex[T number.Number, U any, M metric.Metric[T]](dim uint, leafSize uint, nTrees uint) *bspTreeIndex[T, U, M] {
	return &bspTreeIndex[T, U, M]{
		dim:         dim,
		pool:        make([]*treeElement[T, U], 0),
		roots:       make([]*treeNode[T, U], nTrees),
		leafSize:    leafSize,
		newCutPlane: NewRpCutPlane[T, *treeElement[T, U]],
	}
}

func (bsp *bspTreeIndex[T, U, M]) Add(feature []T, item U) {
	bsp.pool = append(bsp.pool, &treeElement[T, U]{
		Item:    item,
		Feature: feature,
	})
	for i := range bsp.roots {
		bsp.roots[i] = nil
	}
}

func (bsp *bspTreeIndex[T, U, M]) Build() error {
	if len(bsp.pool) == 0 {
		return errors.New("empty pool")
	}

	var buildTree func(elements []*treeElement[T, U]) (*treeNode[T, U], error)
	buildTree = func(elements []*treeElement[T, U]) (*treeNode[T, U], error) {
		if len(elements) == 0 {
			return nil, nil
		}

		if uint(len(elements)) <= bsp.leafSize {
			return &treeNode[T, U]{
				Elements: elements,
			}, nil
		}

		cutPlane, err := bsp.newCutPlane(elements, func(element *treeElement[T, U]) []T {
			return element.Feature
		})
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

	for i := range bsp.roots {
		elements := append([]*treeElement[T, U]{}, bsp.pool...)
		rand.Shuffle(len(elements), func(i, j int) { elements[i], elements[j] = elements[j], elements[i] })

		root, err := buildTree(elements)
		if err != nil {
			return fmt.Errorf("build %d-th index failed", i)
		}
		bsp.roots[i] = root
	}
	return nil
}

func (bsp *bspTreeIndex[T, U, M]) Search(query []T, n uint, r float32) ([]Candidate[U], error) {
	hasIndex := true
	for i := range bsp.roots {
		if bsp.roots[i] == nil {
			hasIndex = false
			break
		}
	}
	if !hasIndex {
		bsp.Build()
	}

	m := 32
	itemQueue := collection.NewUniquePriorityQueue[*U](m)
	nodeQueue := collection.NewPriorityQueue[*treeNode[T, U]](m)
	for i, root := range bsp.roots {
		if root == nil {
			return nil, fmt.Errorf("%d-th index is not created", i)
		}

		nodeQueue.Push(root, float64(math.MaxFloat32))
	}

	for itemQueue.Len() < m && 0 < nodeQueue.Len() {
		node, err := nodeQueue.Pop()
		if err != nil {
			return nil, err
		}
		if node == nil {
			continue
		}

		if node.Left == nil && node.Right == nil {
			for _, element := range node.Elements {
				distance := bsp.metric.CalcDistance(query, element.Feature)
				if distance < r {
					itemQueue.Push(&element.Item, float64(distance))
				}
			}
		} else {
			distanceToCutPlane := node.CutPlane.Distance(query)
			nodeQueue.Push(node.Right, distanceToCutPlane)
			nodeQueue.Push(node.Left, -distanceToCutPlane)
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
		items[i].Distance = float32(item.Priority)
	}

	return items, nil
}
