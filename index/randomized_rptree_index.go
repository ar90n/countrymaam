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

type RandomizedRpTreeIndex[T number.Number, U any, M metric.Metric[T]] struct {
	Dim      uint
	Pool     []*treeElement[T, U]
	Metric   M
	Roots    []*treeNode[T, U]
	LeafSize uint
}

func (rri *RandomizedRpTreeIndex[T, U, M]) Add(feature []T, item U) {
	rri.Pool = append(rri.Pool, &treeElement[T, U]{
		Item:    item,
		Feature: feature,
	})
	for i := range rri.Roots {
		rri.Roots[i] = nil
	}
}

func (rri *RandomizedRpTreeIndex[T, U, M]) Build() error {
	if len(rri.Pool) == 0 {
		return errors.New("empty pool")
	}

	for i := range rri.Roots {
		elements := append([]*treeElement[T, U]{}, rri.Pool...)
		rand.Shuffle(len(elements), func(i, j int) { elements[i], elements[j] = elements[j], elements[i] })

		root, err := buildRpTree(elements, NewRpCutPlane[T, *treeElement[T, U]], rri.LeafSize)
		if err != nil {
			return fmt.Errorf("build %d-th rptree failed", i)
		}
		rri.Roots[i] = root
	}
	return nil
}

func (rri *RandomizedRpTreeIndex[T, U, M]) Search(query []T, n uint, r float32) ([]U, error) {
	hasIndex := true
	for i := range rri.Roots {
		if rri.Roots[i] == nil {
			hasIndex = false
			break
		}
	}
	if !hasIndex {
		rri.Build()
	}

	var search func(elementQueue *collection.UniquePriorityQueue[*treeElement[T, U]], nodeQueue *collection.PriorityQueue[*treeNode[T, U]], node *treeNode[T, U], query []T, metric M, r float32)
	search = func(elementQueue *collection.UniquePriorityQueue[*treeElement[T, U]], nodeQueue *collection.PriorityQueue[*treeNode[T, U]], node *treeNode[T, U], query []T, metric M, r float32) {
		if node == nil {
			return
		}

		if node.Left == nil && node.Right == nil {
			for _, element := range node.Elements {
				distance := metric.CalcDistance(query, element.Feature)
				if distance < r {
					elementQueue.Push(element, float64(distance))
				}
			}
		} else {
			distanceToCutPlane := node.CutPlane.Distance(query)
			primaryNode, secondaryNode := node.Left, node.Right
			if 0.0 < distanceToCutPlane {
				primaryNode, secondaryNode = secondaryNode, primaryNode
			}
			search(elementQueue, nodeQueue, primaryNode, query, metric, r)
			nodeQueue.Push(secondaryNode, math.Abs(distanceToCutPlane))
		}
	}

	m := 32
	elementQueue := collection.NewUniquePriorityQueue[*treeElement[T, U]](m)
	nodeQueue := collection.NewPriorityQueue[*treeNode[T, U]](m)
	for i, root := range rri.Roots {
		if root == nil {
			return nil, fmt.Errorf("%d-th index is not created", i)
		}

		search(elementQueue, nodeQueue, root, query, rri.Metric, r)
	}

	for elementQueue.Len() < m && 0 < nodeQueue.Len() {
		node, err := nodeQueue.Pop()
		if err != nil {
			return nil, err
		}
		search(elementQueue, nodeQueue, node, query, rri.Metric, r)
	}

	n = number.Min(n, uint(elementQueue.Len()))
	items := make([]U, n)
	for i := uint(0); i < n; i++ {
		element, err := elementQueue.Pop()
		if err != nil {
			return nil, err
		}
		items[i] = element.Item
	}

	return items, nil
}

func NewRandomizedRpTreeIndex[T number.Number, U any, M metric.Metric[T]](dim uint, leafSize uint, nTrees uint) *RandomizedRpTreeIndex[T, U, M] {
	return &RandomizedRpTreeIndex[T, U, M]{
		Dim:      dim,
		Pool:     make([]*treeElement[T, U], 0),
		Roots:    make([]*treeNode[T, U], nTrees),
		LeafSize: leafSize,
	}
}
