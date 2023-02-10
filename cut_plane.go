package countrymaam

import (
	"errors"
	"math"
	"math/rand"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
)

type CutPlaneFactory[T linalg.Number, U comparable] interface {
	Default() CutPlane[T, U]
	Build(elements []treeElement[T, U], indice []int, env linalg.Env[T]) (CutPlane[T, U], error)
}

type CutPlane[T linalg.Number, U comparable] interface {
	Evaluate(feature []T, env linalg.Env[T]) bool
	Distance(feature []T, env linalg.Env[T]) float64
}

// kdCutPlane is a cut plane that is constructed by kdtree algorithm.
// this is derived from flann library.
// https://github.com/flann-lib/flann/blob/master/src/cpp/flann/algorithms/kdtree_index.h
type kdCutPlane[T linalg.Number, U comparable] struct {
	Axis  uint
	Value float64
}

func (cp kdCutPlane[T, U]) Evaluate(feature []T, env linalg.Env[T]) bool {
	return 0.0 <= cp.Distance(feature, env)
}

func (cp kdCutPlane[T, U]) Distance(feature []T, env linalg.Env[T]) float64 {
	return float64(feature[cp.Axis]) - cp.Value
}

type rpCutPlane[T linalg.Number, U comparable] struct {
	NormalVector []float32
	A            float64
}

func (cp rpCutPlane[T, U]) Evaluate(feature []T, env linalg.Env[T]) bool {
	return 0.0 <= cp.Distance(feature, env)
}

func (cp rpCutPlane[T, U]) Distance(feature []T, env linalg.Env[T]) float64 {
	return cp.A + float64(env.DotWithF32(feature, cp.NormalVector))
}

type kdCutPlaneFactory[T linalg.Number, U comparable] struct {
	features   uint
	candidates uint
}

func NewKdCutPlaneFactory[T linalg.Number, U comparable](features, candidates uint) CutPlaneFactory[T, U] {
	return kdCutPlaneFactory[T, U]{
		features:   features,
		candidates: candidates,
	}
}

func (f kdCutPlaneFactory[T, U]) Default() CutPlane[T, U] {
	return kdCutPlane[T, U]{}
}

func (f kdCutPlaneFactory[T, U]) Build(elements []treeElement[T, U], indice []int, env linalg.Env[T]) (CutPlane[T, U], error) {
	if len(indice) == 0 {
		return nil, errors.New("elements is empty")
	}

	dim := len(elements[0].Feature)
	accs := make([]float64, dim)
	sqAccs := make([]float64, dim)
	nSamples := uint(len(indice))
	if 0 < f.features && f.features < nSamples {
		nSamples = f.features
	}
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
			Value: mean,
		}
		queue.Push(cutPlane, -float64(variance))
	}

	// Randomly select one of the best candidates.
	nCandidates := linalg.Min(int(f.candidates), queue.Len())
	nSkip := 0
	if 0 < nCandidates {
		nSkip = rand.Intn(nCandidates) - 1
	}
	for i := 0; i < nSkip; i++ {
		queue.Pop()
	}
	return queue.Pop()
}

type rpCutPlaneFactory[T linalg.Number, U comparable] struct {
	features uint
}

func NewRpCutPlaneFactory[T linalg.Number, U comparable](features uint) CutPlaneFactory[T, U] {
	return rpCutPlaneFactory[T, U]{features: features}
}

func (f rpCutPlaneFactory[T, U]) Default() CutPlane[T, U] {
	return rpCutPlane[T, U]{}
}

func (f rpCutPlaneFactory[T, U]) Build(elements []treeElement[T, U], indice []int, env linalg.Env[T]) (CutPlane[T, U], error) {
	if len(indice) == 0 {
		return nil, errors.New("elements is empty")
	}

	lhsIndex := rand.Intn(len(indice))
	rhsIndex := rand.Intn(len(indice) - 1)
	if lhsIndex <= rhsIndex {
		rhsIndex++
	}

	const maxIter = 8
	dim := len(elements[indice[lhsIndex]].Feature)
	lhsCenter := make([]float32, dim)
	rhsCenter := make([]float32, dim)
	lhsCount := 1
	rhsCount := 1
	for i := 0; i < dim; i++ {
		lhsCenter[i] = float32(elements[indice[lhsIndex]].Feature[i])
		rhsCenter[i] = float32(elements[indice[rhsIndex]].Feature[i])
	}
	nSamples := uint(32)
	if 0 < f.features {
		nSamples = f.features
	}
	if uint(len(indice)) < nSamples {
		nSamples = uint(len(indice))
	}

	for i := 0; i < maxIter; i++ {
		rand.Shuffle(len(indice), func(i, j int) { indice[i], indice[j] = indice[j], indice[i] })
		for _, k := range indice[:nSamples] {
			feature := elements[k].Feature
			lhsSqDist := env.SqL2WithF32(feature, lhsCenter)
			rhsSqDist := env.SqL2WithF32(feature, rhsCenter)

			if lhsSqDist < rhsSqDist {
				invCountPlusOone := 1.0 / float32(lhsCount+1)
				for j, v := range feature {
					lhsCenter[j] = (lhsCenter[j]*float32(lhsCount) + float32(v)) * invCountPlusOone
				}
				lhsCount++
			} else {
				invCountPlusOone := 1.0 / float32(rhsCount+1)
				for j, v := range feature {
					rhsCenter[j] = (rhsCenter[j]*float32(rhsCount) + float32(v)) * invCountPlusOone
				}
				rhsCount++
			}
		}
	}

	accSqDiff := float32(0.0)
	normalVector := make([]float32, dim)
	for i := 0; i < dim; i++ {
		diff := lhsCenter[i] - rhsCenter[i]
		normalVector[i] = diff
		accSqDiff += diff * diff
	}
	invNorm := 1.0 / (math.Sqrt(float64(accSqDiff)) + 1e-10)
	for i := 0; i < dim; i++ {
		normalVector[i] *= float32(invNorm)
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
