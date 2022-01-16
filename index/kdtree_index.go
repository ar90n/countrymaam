package index

import (
	"errors"
	"sort"

	"github.com/ar90n/countrymaam/collection"
	my_constraints "github.com/ar90n/countrymaam/constraints"
	"github.com/ar90n/countrymaam/metric"
)

type kdCutPlane[T my_constraints.Number, U any] struct {
	Axis  uint
	Value T
}

func (cp kdCutPlane[T, U]) Evaluate(element *kdElement[T, U]) bool {
	return element.Feature[cp.Axis] <= cp.Value
}

func NewKdCutPlane[T my_constraints.Number, U any](elements []*kdElement[T, U]) (kdCutPlane[T, U], error) {
	if len(elements) == 0 {
		return kdCutPlane[T, U]{}, errors.New("elements is empty")
	}

	minValues := append([]T{}, elements[0].Feature...)
	maxValues := append([]T{}, elements[0].Feature...)
	for _, element := range elements[1:] {
		for j, v := range element.Feature {
			minValues[j] = my_constraints.Min(minValues[j], v)
			maxValues[j] = my_constraints.Max(maxValues[j], v)
		}
	}

	maxRange := maxValues[0] - minValues[0]
	cutPlane := kdCutPlane[T, U]{
		Axis:  uint(0),
		Value: (maxValues[0] + minValues[0]) / 2,
	}
	for i := uint(1); i < uint(len(minValues)); i++ {
		diff := maxValues[i] - minValues[i]
		if maxRange < diff {
			maxRange = diff
			cutPlane = kdCutPlane[T, U]{
				Axis:  i,
				Value: (maxValues[i] + minValues[i]) / 2,
			}
		}
	}

	return cutPlane, nil
}

type kdElement[T my_constraints.Number, U any] struct {
	Feature []T
	Item    U
}

type kdNode[T my_constraints.Number, U any] struct {
	CutPlane kdCutPlane[T, U]
	Elements []*kdElement[T, U]
	Left     *kdNode[T, U]
	Right    *kdNode[T, U]
}

type kdTreeIndex[T my_constraints.Number, U any, M metric.Metric[T]] struct {
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

	var search func(node *kdNode[T, U], query []T, metric M, n uint, r float32) []Candidate[U]
	search = func(node *kdNode[T, U], query []T, metric M, n uint, r float32) []Candidate[U] {
		if node == nil {
			return []Candidate[U]{}
		}

		candidates := make([]Candidate[U], 0)
		if node.Left == nil && node.Right == nil {
			for _, element := range node.Elements {
				distance := metric.CalcDistance(query, element.Feature)
				if distance < r {
					candidates = append(candidates, Candidate[U]{distance, element.Item})
				}
			}
		} else {
			primaryNode, secondaryNode := node.Left, node.Right
			if node.CutPlane.Value < query[node.CutPlane.Axis] {
				primaryNode, secondaryNode = secondaryNode, primaryNode
			}
			candidates = append(candidates, search(primaryNode, query, metric, n, r)...)

			if n < uint(len(candidates)) {
				r = candidates[len(candidates)-1].Distance
			}

			distanceToCutPlane := float32(my_constraints.Abs(query[node.CutPlane.Axis] - node.CutPlane.Value))
			if distanceToCutPlane < r {
				candidates = append(candidates, search(secondaryNode, query, metric, n, r)...)
			}
		}

		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Distance < candidates[j].Distance
		})
		if n < uint(len(candidates)) {
			candidates = candidates[:n]
		}
		return candidates
	}

	candidates := search(ki.Root, query, ki.Metric, n, r)
	results := make([]U, len(candidates))
	for i, c := range candidates {
		results[i] = c.Item
	}
	return results, nil
}

func (ki *kdTreeIndex[T, U, M]) Build() error {
	var build func(elements []*kdElement[T, U], leafSize uint) (*kdNode[T, U], error)
	build = func(elements []*kdElement[T, U], leafSize uint) (*kdNode[T, U], error) {
		if len(elements) == 0 {
			return nil, nil
		}

		if uint(len(elements)) <= leafSize {
			return &kdNode[T, U]{
				Elements: elements,
			}, nil
		}

		cutPlane, err := NewKdCutPlane(elements)
		if err != nil {
			return nil, err
		}

		leftElements, rightElements := collection.Partition(elements, cutPlane)
		left, err := build(leftElements, leafSize)
		if err != nil {
			return nil, err
		}
		right, err := build(rightElements, leafSize)
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

	if len(ki.Pool) == 0 {
		return errors.New("empty pool")
	}

	root, err := build(ki.Pool, ki.LeafSize)
	if err != nil {
		return errors.New("build failed")
	}
	ki.Root = root
	return nil
}

func NewKDTreeIndex[T my_constraints.Number, U any, M metric.Metric[T]](dim, leafSize uint) *kdTreeIndex[T, U, M] {
	return &kdTreeIndex[T, U, M]{
		Dim:      dim,
		LeafSize: leafSize,
	}
}
