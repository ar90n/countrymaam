package index

import (
	"errors"
	"math"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/metric"
	"github.com/ar90n/countrymaam/number"
)

type rpTreeIndex[T number.Number, U any, M metric.Metric[T]] struct {
	Dim      uint
	Pool     []*treeElement[T, U]
	Metric   M
	Root    *treeNode[T, U]
	LeafSize uint
}

func (ri *rpTreeIndex[T, U, M]) Add(feature []T, item U) {
	ri.Pool = append(ri.Pool, &treeElement[T, U]{
		Feature: feature,
		Item:    item,
	})
	if ri.Root != nil {
		ri.Root = nil
	}
}

func (ri rpTreeIndex[T, U, M]) Search(query []T, n uint, r float32) ([]U, error) {
	if ri.Root == nil {
		ri.Build()
	}
	if ri.Root == nil {
		return nil, errors.New("index is not created")
	}

	var search func(queue *collection.PriorityQueue[U], node *treeNode[T, U], query []T, metric M, n uint, r float32)
	search = func(queue *collection.PriorityQueue[U], node *treeNode[T, U], query []T, metric M, n uint, r float32) {
		if node == nil {
			return
		}

		if node.Left == nil && node.Right == nil {
			for _, element := range node.Elements {
				distance := metric.CalcDistance(query, element.Feature)
				if distance < r {
					queue.Push(element.Item, float64(distance))
				}
			}
		} else {
			distanceToCutPlane := node.CutPlane.Distance(query)
			primaryNode, secondaryNode := node.Left, node.Right
			if 0.0 < distanceToCutPlane {
				primaryNode, secondaryNode = secondaryNode, primaryNode
			}
			search(queue, primaryNode, query, metric, n, r)

			item, err := queue.PeekWithPriority(int(n) - 1)
			if err == nil {
				r = float32(item.Priority)
			}

			if math.Abs(distanceToCutPlane) < float64(r) {
				search(queue, secondaryNode, query, metric, n, r)
			}
		}
	}

	queue := collection.PriorityQueue[U]{}
	search(&queue, ri.Root, query, ri.Metric, n, r)

	results := make([]U, number.Min(n, uint(queue.Len())))
	for i := range results {
		item, err := queue.Pop()
		if err != nil {
			return nil, err
		}
		results[i] = item
	}
	return results, nil
}

func buildRpTree[T number.Number, U any](elements []*treeElement[T, U], makeCutPlane CutPlaneConstructor[T, *treeElement[T, U]], leafSize uint) (*treeNode[T, U], error) {
	if len(elements) == 0 {
		return nil, nil
	}

	if uint(len(elements)) <= leafSize {
		return &treeNode[T, U]{
			Elements: elements,
		}, nil
	}

	cutPlane, err := makeCutPlane(elements, func(element *treeElement[T, U]) []T {
		return element.Feature
	})
	if err != nil {
		return nil, err
	}

	leftElements, rightElements := collection.Partition(elements,
		func(element *treeElement[T, U]) bool {
			return cutPlane.Evaluate(element.Feature)
		})
	left, err := buildKdTree(leftElements, makeCutPlane, leafSize)
	if err != nil {
		return nil, err
	}
	right, err := buildKdTree(rightElements, makeCutPlane, leafSize)
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

func (ri *rpTreeIndex[T, U, M]) Build() error {
	if len(ri.Pool) == 0 {
		return errors.New("empty pool")
	}

	root, err := buildRpTree(ri.Pool, NewRpCutPlane[T, *treeElement[T,U]], ri.LeafSize)
	if err != nil {
		return errors.New("build failed")
	}
	ri.Root = root
	return nil
}

func NewRpTreeIndex[T number.Number, U any, M metric.Metric[T]](dim, leafSize uint) *rpTreeIndex[T, U, M] {
	return &rpTreeIndex[T, U, M]{
		Dim:      dim,
		LeafSize: leafSize,
	}
}