package index

import (
	"errors"
	"math"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/metric"
	"github.com/ar90n/countrymaam/number"
)

type kdElement[T number.Number, U any] struct {
	Feature []T
	Item    U
}

type kdNode[T number.Number, U any] struct {
	CutPlane kdCutPlane[T, U]
	Elements []*kdElement[T, U]
	Left     *kdNode[T, U]
	Right    *kdNode[T, U]
}

type kdTreeIndex[T number.Number, U any, M metric.Metric[T]] struct {
	Dim      uint
	Pool     []*kdElement[T, U]
	Metric   M
	Root     *kdNode[T, U]
	LeafSize uint
}

var _ = (*kdTreeIndex[float32, int, metric.SqL2Dist[float32]])(nil)

func (ki *kdTreeIndex[T, U, M]) Add(feature []T, item U) {
	ki.Pool = append(ki.Pool, &kdElement[T, U]{
		Feature: feature,
		Item:    item,
	})
	if ki.Root != nil {
		ki.Root = nil
	}
}

func (ki kdTreeIndex[T, U, M]) Search(query []T, n uint, r float32) ([]U, error) {
	if ki.Root == nil {
		ki.Build()
	}
	if ki.Root == nil {
		return nil, errors.New("index is not created")
	}

	var search func(queue *collection.PriorityQueue[U], node *kdNode[T, U], query []T, metric M, n uint, r float32)
	search = func(queue *collection.PriorityQueue[U], node *kdNode[T, U], query []T, metric M, n uint, r float32) {
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
	search(&queue, ki.Root, query, ki.Metric, n, r)

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

type CutPlaneConstructor[T number.Number, U any] func(elements []*kdElement[T, U]) (kdCutPlane[T, U], error)

func buildKdTree[T number.Number, U any](elements []*kdElement[T, U], makeCutPlane CutPlaneConstructor[T, U], leafSize uint) (*kdNode[T, U], error) {
	if len(elements) == 0 {
		return nil, nil
	}

	if uint(len(elements)) <= leafSize {
		return &kdNode[T, U]{
			Elements: elements,
		}, nil
	}

	cutPlane, err := makeCutPlane(elements)
	if err != nil {
		return nil, err
	}

	leftElements, rightElements := collection.Partition(elements,
		func(element *kdElement[T, U]) bool {
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

	return &kdNode[T, U]{
		Elements: elements,
		Left:     left,
		Right:    right,
		CutPlane: cutPlane,
	}, nil
}

func (ki *kdTreeIndex[T, U, M]) Build() error {
	if len(ki.Pool) == 0 {
		return errors.New("empty pool")
	}

	root, err := buildKdTree(ki.Pool, NewKdCutPlane[T, U], ki.LeafSize)
	if err != nil {
		return errors.New("build failed")
	}
	ki.Root = root
	return nil
}

func NewKDTreeIndex[T number.Number, U any, M metric.Metric[T]](dim, leafSize uint) *kdTreeIndex[T, U, M] {
	return &kdTreeIndex[T, U, M]{
		Dim:      dim,
		LeafSize: leafSize,
	}
}
