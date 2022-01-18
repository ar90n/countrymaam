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

func NewRandomizedKdCutPlane[T number.Number, U any](elements []*kdElement[T, U]) (kdCutPlane[T, U], error) {
	if len(elements) == 0 {
		return kdCutPlane[T, U]{}, errors.New("elements is empty")
	}

	dim := len(elements[0].Feature)
	accs := make([]float64, dim)
	sqAccs := make([]float64, dim)
	for _, element := range elements {
		for j, v := range element.Feature {
			v := float64(v)
			accs[j] += v
			sqAccs[j] += v * v
		}
	}

	invN := 1.0 / float64(len(elements))
	queue := collection.PriorityQueue[kdCutPlane[T, U]]{}
	for i := range accs {
		mean := accs[i] * invN
		sqMean := sqAccs[i] * invN
		variance := sqMean - mean*mean

		cutPlane := kdCutPlane[T, U]{
			Axis:  uint(i),
			Value: number.Cast[float64, T](mean),
		}
		queue.Push(cutPlane, -variance)
	}

	nCandidates := number.Min(5, queue.Len())
	nSkip := rand.Intn(nCandidates) - 1
	for i := 0; i < nSkip; i++ {
		queue.Pop()
	}
	return queue.Pop()
}

type RandomizedKdTreeIndex[T number.Number, U any, M metric.Metric[T]] struct {
	Dim      uint
	Pool     []*kdElement[T, U]
	Metric   M
	Roots    []*kdNode[T, U]
	LeafSize uint
}

func (rki *RandomizedKdTreeIndex[T, U, M]) Add(feature []T, item U) {
	rki.Pool = append(rki.Pool, &kdElement[T, U]{
		Item:    item,
		Feature: feature,
	})
	for i := range rki.Roots {
		rki.Roots[i] = nil
	}
}

func (rki *RandomizedKdTreeIndex[T, U, M]) Build() error {
	if len(rki.Pool) == 0 {
		return errors.New("empty pool")
	}

	for i := range rki.Roots {
		elements := append([]*kdElement[T, U]{}, rki.Pool...)
		rand.Shuffle(len(elements), func(i, j int) { elements[i], elements[j] = elements[j], elements[i] })

		root, err := buildKdTree(elements, rki.LeafSize)
		if err != nil {
			return fmt.Errorf("build %d-th kdtree failed", i)
		}
		rki.Roots[i] = root
	}
	return nil
}

func (rki *RandomizedKdTreeIndex[T, U, M]) Search(query []T, n uint, r float32) ([]U, error) {
	hasIndex := true
	for i := range rki.Roots {
		if rki.Roots[i] == nil {
			hasIndex = false
			break
		}
	}
	if !hasIndex {
		rki.Build()
	}

	var search func(elementQueue *collection.UniquePriorityQueue[*kdElement[T, U]], nodeQueue *collection.PriorityQueue[*kdNode[T, U]], node *kdNode[T, U], query []T, metric M, r float32)
	search = func(elementQueue *collection.UniquePriorityQueue[*kdElement[T, U]], nodeQueue *collection.PriorityQueue[*kdNode[T, U]], node *kdNode[T, U], query []T, metric M, r float32) {
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
	elementQueue := collection.NewUniquePriorityQueue[*kdElement[T, U]](m)
	nodeQueue := collection.NewPriorityQueue[*kdNode[T, U]](m)
	for i, root := range rki.Roots {
		if root == nil {
			return nil, fmt.Errorf("%d-th index is not created", i)
		}

		search(elementQueue, nodeQueue, root, query, rki.Metric, r)
	}

	for elementQueue.Len() < m && 0 < nodeQueue.Len() {
		node, err := nodeQueue.Pop()
		if err != nil {
			return nil, err
		}
		search(elementQueue, nodeQueue, node, query, rki.Metric, r)
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

func NewRandomizedKdTreeIndex[T number.Number, U any, M metric.Metric[T]](dim uint, leafSize uint, nTrees uint) *RandomizedKdTreeIndex[T, U, M] {
	return &RandomizedKdTreeIndex[T, U, M]{
		Dim:      dim,
		Pool:     make([]*kdElement[T, U], 0),
		Roots:    make([]*kdNode[T, U], nTrees),
		LeafSize: leafSize,
	}
}
