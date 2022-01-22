package countrymaam

import (
	"errors"
	"math"
	"math/rand"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/number"
)

type CutPlane[T number.Number, U any] interface {
	Evaluate(feature []T) bool
	Distance(feature []T) float64
	Construct(features []*treeElement[T, U]) (CutPlane[T, U], error)
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

func (cp kdCutPlane[T, U]) Construct(elements []*treeElement[T, U]) (CutPlane[T, U], error) {
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

	return &cutPlane, nil
}

type randomizedKdCutPlane[T number.Number, U any] struct {
	Axis  uint
	Value T
}

func (cp randomizedKdCutPlane[T, U]) Evaluate(feature []T) bool {
	return 0.0 <= cp.Distance(feature)
}

func (cp randomizedKdCutPlane[T, U]) Distance(feature []T) float64 {
	return float64(feature[cp.Axis] - cp.Value)
}

func (cp randomizedKdCutPlane[T, U]) Construct(elements []*treeElement[T, U]) (CutPlane[T, U], error) {
	if len(elements) == 0 {
		return nil, errors.New("elements is empty")
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
	queue := collection.PriorityQueue[*kdCutPlane[T, U]]{}
	for i := range accs {
		mean := accs[i] * invN
		sqMean := sqAccs[i] * invN
		variance := sqMean - mean*mean

		cutPlane := &kdCutPlane[T, U]{
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

type rpCutPlane[T number.Number, U any] struct {
	NormalVector []T
	A            T
}

func (cp rpCutPlane[T, U]) Evaluate(feature []T) bool {
	return 0.0 <= cp.Distance(feature)
}

func (cp rpCutPlane[T, U]) Distance(feature []T) float64 {
	dot := float64(cp.A)
	for i := range feature {
		dot += float64(feature[i]) * float64(cp.NormalVector[i])
	}
	return dot
}

func (cp rpCutPlane[T, U]) Construct(elements []*treeElement[T, U]) (CutPlane[T, U], error) {
	if len(elements) == 0 {
		return nil, errors.New("elements is empty")
	}

	lhsIndex := rand.Intn(len(elements))
	rhsIndex := rand.Intn(len(elements) - 1)
	if lhsIndex <= rhsIndex {
		rhsIndex++
	}

	maxIter := 200
	dim := len(elements[lhsIndex].Feature)
	lhsCenter := make([]float64, dim)
	rhsCenter := make([]float64, dim)
	lhsCount := 1
	rhsCount := 1
	for i := 0; i < dim; i++ {
		lhsCenter[i] = float64(elements[lhsIndex].Feature[i])
		rhsCenter[i] = float64(elements[rhsIndex].Feature[i])
	}
	for i := 0; i < maxIter; i++ {
		for _, element := range elements {
			lhsSqDist := 0.0
			rhsSqDist := 0.0
			feature := element.Feature
			for j := range feature {
				lhsDiff := float64(lhsCenter[j]) - float64(feature[j])
				lhsSqDist += lhsDiff * lhsDiff
				rhsDiff := float64(rhsCenter[j]) - float64(feature[j])
				rhsSqDist += rhsDiff * rhsDiff
			}

			if lhsSqDist < rhsSqDist {
				for j, v := range feature {
					lhsCenter[j] = (lhsCenter[j]*float64(lhsCount) + float64(v)) / float64(lhsCount+1)
				}
				lhsCount++
			} else {
				for j, v := range feature {
					rhsCenter[j] = (rhsCenter[j]*float64(rhsCount) + float64(v)) / float64(rhsCount+1)
				}
				rhsCount++
			}
		}
	}

	accSqDiff := 0.0
	normalVector := make([]T, dim)
	for i := 0; i < dim; i++ {
		diff := float64(lhsCenter[i] - rhsCenter[i])
		normalVector[i] = number.Cast[float64, T](diff)
		accSqDiff += diff
	}
	norm := math.Sqrt(accSqDiff)
	for i := 0; i < dim; i++ {
		normalVector[i] /= number.Cast[float64, T](norm)
	}
	fa := 0.0
	for i := 0; i < dim; i++ {
		fa -= float64(normalVector[i]) * float64(rhsCenter[i]+lhsCenter[i]) / 2.0
	}
	a := number.Cast[float64, T](fa)

	cutPlane := rpCutPlane[T, U]{
		NormalVector: normalVector,
		A:            a,
	}
	return &cutPlane, nil
}
