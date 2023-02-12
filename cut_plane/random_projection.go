package cut_plane

import (
	"errors"
	"math"
	"math/rand"

	"github.com/ar90n/countrymaam/index"
	"github.com/ar90n/countrymaam/linalg"
)

type RpCutPlane[T linalg.Number, U comparable] struct {
	normal []float32
	a      float64
}

func (cp RpCutPlane[T, U]) Evaluate(feature []T, env linalg.Env[T]) bool {
	return 0.0 <= cp.Distance(feature, env)
}

func (cp RpCutPlane[T, U]) Distance(feature []T, env linalg.Env[T]) float64 {
	return cp.a + float64(env.DotWithF32(feature, cp.normal))
}

type RpCutPlaneFactory[T linalg.Number, U comparable] struct {
	features uint
}

func NewRpCutPlaneFactory[T linalg.Number, U comparable](features uint) index.CutPlaneFactory[T, U] {
	return RpCutPlaneFactory[T, U]{features: features}
}

func (f RpCutPlaneFactory[T, U]) Default() index.CutPlane[T, U] {
	return RpCutPlane[T, U]{}
}

func (f RpCutPlaneFactory[T, U]) Build(elements []index.TreeElement[T, U], indice []int, env linalg.Env[T]) (index.CutPlane[T, U], error) {
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
	normal := make([]float32, dim)
	for i := 0; i < dim; i++ {
		diff := lhsCenter[i] - rhsCenter[i]
		normal[i] = diff
		accSqDiff += diff * diff
	}
	invNorm := 1.0 / (math.Sqrt(float64(accSqDiff)) + 1e-10)
	for i := 0; i < dim; i++ {
		normal[i] *= float32(invNorm)
	}

	a := 0.0
	for i := 0; i < dim; i++ {
		a -= float64(normal[i]) * float64(rhsCenter[i]+lhsCenter[i])
	}
	a /= 2.0

	cutPlane := RpCutPlane[T, U]{
		normal: normal,
		a:      a,
	}
	return &cutPlane, nil
}
