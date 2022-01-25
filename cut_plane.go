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
	Construct(elements []treeElement[T, U], indice []int) (CutPlane[T, U], error)
}

type kdCutPlane[T number.Number, U any] struct {
	Axis  uint
	Value T
}

func (cp kdCutPlane[T, U]) Evaluate(feature []T) bool {
	return 0.0 <= cp.Distance(feature)
}

func (cp kdCutPlane[T, U]) Distance(feature []T) float64 {
	diff := float64(feature[cp.Axis] - cp.Value)
	sqDist := diff * diff
	if diff < 0.0 {
		sqDist = -sqDist
	}
	return sqDist
}

func (cp kdCutPlane[T, U]) Construct(elements []treeElement[T, U], indice []int) (CutPlane[T, U], error) {
	minValues := append([]T{}, elements[indice[0]].Feature...)
	maxValues := append([]T{}, elements[indice[0]].Feature...)
	for _, i := range indice {
		element := elements[i]
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
	diff := float64(feature[cp.Axis] - cp.Value)
	sqDist := diff * diff
	if diff < 0.0 {
		sqDist = -sqDist
	}
	return sqDist
}

func (cp randomizedKdCutPlane[T, U]) Construct(elements []treeElement[T, U], indice []int) (CutPlane[T, U], error) {
	if len(indice) == 0 {
		return nil, errors.New("elements is empty")
	}

	dim := len(elements[0].Feature)
	accs := make([]float64, dim)
	sqAccs := make([]float64, dim)
	nSamples := number.Min(uint(len(indice)), 100)
	for _, i := range indice[:nSamples] {
		element := elements[i]
		for j, v := range element.Feature {
			v := float64(v)
			accs[j] += v
			sqAccs[j] += v * v
		}
	}

	invN := 1.0 / float64(nSamples)
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
	NormalVector []float64
	A            float64
}

func (cp rpCutPlane[T, U]) Evaluate(feature []T) bool {
	return 0.0 <= cp.Distance(feature)
}

func (cp rpCutPlane[T, U]) Distance(feature []T) float64 {
	dot := cp.A + number.CalcDot(feature, cp.NormalVector)
	sqDist := dot * dot
	if dot < 0.0 {
		sqDist = -sqDist
	}

	return sqDist
}

func (cp rpCutPlane[T, U]) Construct(elements []treeElement[T, U], indice []int) (CutPlane[T, U], error) {
	if len(indice) == 0 {
		return nil, errors.New("elements is empty")
	}

	lhsIndex := rand.Intn(len(indice))
	rhsIndex := rand.Intn(len(indice) - 1)
	if lhsIndex <= rhsIndex {
		rhsIndex++
	}

	const maxIter = 32
	dim := len(elements[indice[lhsIndex]].Feature)
	lhsCenter := make([]float64, dim)
	rhsCenter := make([]float64, dim)
	lhsCount := 1
	rhsCount := 1
	for i := 0; i < dim; i++ {
		lhsCenter[i] = float64(elements[indice[lhsIndex]].Feature[i])
		rhsCenter[i] = float64(elements[indice[rhsIndex]].Feature[i])
	}
	nSamples := number.Min(uint(len(indice)), 32)
	for i := 0; i < maxIter; i++ {
		rand.Shuffle(len(indice), func(i, j int) { indice[i], indice[j] = indice[j], indice[i] })
		for _, k := range indice[:nSamples] {
			feature := elements[k].Feature
			lhsSqDist := number.CalcSqDist(feature, lhsCenter)
			rhsSqDist := number.CalcSqDist(feature, rhsCenter)

			if lhsSqDist < rhsSqDist {
				invCountPlusOone := 1.0 / float64(lhsCount+1)
				for j, v := range feature {
					lhsCenter[j] = (lhsCenter[j]*float64(lhsCount) + float64(v)) * invCountPlusOone
				}
				lhsCount++
			} else {
				invCountPlusOone := 1.0 / float64(rhsCount+1)
				for j, v := range feature {
					rhsCenter[j] = (rhsCenter[j]*float64(rhsCount) + float64(v)) * invCountPlusOone
				}
				rhsCount++
			}
		}
	}

	accSqDiff := 0.0
	normalVector := make([]float64, dim)
	for i := 0; i < dim; i++ {
		diff := lhsCenter[i] - rhsCenter[i]
		normalVector[i] = diff
		accSqDiff += diff * diff
	}
	invNorm := 1.0 / (math.Sqrt(accSqDiff) + 1e-10)
	for i := 0; i < dim; i++ {
		normalVector[i] *= invNorm
	}

	a := 0.0
	for i := 0; i < dim; i++ {
		a -= float64(normalVector[i]) * float64(rhsCenter[i]+lhsCenter[i])
	}
	a /= 2.0

	cutPlane := rpCutPlane[T, U]{
		NormalVector: normalVector,
		A:            a,
	}
	return &cutPlane, nil
}
