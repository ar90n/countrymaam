package index

import (
	"errors"

	"github.com/ar90n/countrymaam/number"
)

type CutPlane[T number.Number] interface {
	Evaluate(feature []T) bool
	Distance(feature []T) float64
}

type kdCutPlane[T number.Number, U any] struct {
	Axis  uint
	Value T
}

func (cp kdCutPlane[T, U]) Evaluate(feature []T) bool {
	return 0.0 <= cp.Distance(feature)
}

func (cp kdCutPlane[T, U]) Distance(feature []T) float64 {
	return float64(feature[cp.Axis] - cp.Value)
}

func NewKdCutPlane[T number.Number, U any](elements []*kdElement[T, U]) (kdCutPlane[T, U], error) {
	if len(elements) == 0 {
		return kdCutPlane[T, U]{}, errors.New("elements is empty")
	}

	minValues := append([]T{}, elements[0].Feature...)
	maxValues := append([]T{}, elements[0].Feature...)
	for _, element := range elements[1:] {
		for j, v := range element.Feature {
			minValues[j] = number.Min(minValues[j], v)
			maxValues[j] = number.Max(maxValues[j], v)
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
