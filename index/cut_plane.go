package index

import (
	"errors"
	"math/rand"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/number"
)

type CutPlaneConstructor[T number.Number, U any] func(elements []U, selector func(element U) []T) (CutPlane[T], error)

type CutPlane[T number.Number] interface {
	Evaluate(feature []T) bool
	Distance(feature []T) float64
}

type kdCutPlane[T number.Number] struct {
	Axis  uint
	Value T
}

func (cp kdCutPlane[T]) Evaluate(feature []T) bool {
	return 0.0 <= cp.Distance(feature)
}

func (cp kdCutPlane[T]) Distance(feature []T) float64 {
	return float64(feature[cp.Axis] - cp.Value)
}

func NewKdCutPlane[T number.Number, U any](elements []U, selector func(element U) []T) (CutPlane[T], error) {
	if len(elements) == 0 {
		return nil, errors.New("elements is empty")
	}

	minValues := append([]T{}, selector(elements[0])...)
	maxValues := append([]T{}, selector(elements[0])...)
	for _, element := range elements[1:] {
		for j, v := range selector(element) {
			minValues[j] = number.Min(minValues[j], v)
			maxValues[j] = number.Max(maxValues[j], v)
		}
	}

	maxRange := maxValues[0] - minValues[0]
	cutPlane := kdCutPlane[T]{
		Axis:  uint(0),
		Value: (maxValues[0] + minValues[0]) / 2,
	}
	for i := uint(1); i < uint(len(minValues)); i++ {
		diff := maxValues[i] - minValues[i]
		if maxRange < diff {
			maxRange = diff
			cutPlane = kdCutPlane[T]{
				Axis:  i,
				Value: (maxValues[i] + minValues[i]) / 2,
			}
		}
	}

	return &cutPlane, nil
}

func NewRandomizedKdCutPlane[T number.Number, U any](elements []U, selector func(element U) []T) (CutPlane[T], error) {
	if len(elements) == 0 {
		return nil, errors.New("elements is empty")
	}

	dim := len(selector(elements[0]))
	accs := make([]float64, dim)
	sqAccs := make([]float64, dim)
	for _, element := range elements {
		for j, v := range selector(element) {
			v := float64(v)
			accs[j] += v
			sqAccs[j] += v * v
		}
	}

	invN := 1.0 / float64(len(elements))
	queue := collection.PriorityQueue[*kdCutPlane[T]]{}
	for i := range accs {
		mean := accs[i] * invN
		sqMean := sqAccs[i] * invN
		variance := sqMean - mean*mean

		cutPlane := &kdCutPlane[T]{
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
