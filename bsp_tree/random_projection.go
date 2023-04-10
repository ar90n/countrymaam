package bsp_tree

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/ar90n/countrymaam/linalg"
)

const (
	rpDefaultLeafs          = 16
	rpDefaultSampleFeatures = 32
)

var (
	_ CutPlane[float32] = (*rpCutPlane[float32])(nil)
)

type rpCutPlane[T linalg.Number] struct {
	Normal []float32
	A      float64
}

func (cp rpCutPlane[T]) Evaluate(feature []T, env linalg.Env[T]) bool {
	return 0.0 <= cp.Distance(feature, env)
}

func (cp rpCutPlane[T]) Distance(feature []T, env linalg.Env[T]) float64 {
	return cp.A + float64(env.DotWithF32(feature, cp.Normal))
}

func newRpCutPlane[T linalg.Number](features [][]T, indice []int, sampleFeatures uint, env linalg.Env[T]) (CutPlane[T], error) {
	if len(indice) == 0 {
		return nil, errors.New("elements is empty")
	}

	lhsIndex := rand.Intn(len(indice))
	rhsIndex := rand.Intn(len(indice) - 1)
	if lhsIndex <= rhsIndex {
		rhsIndex++
	}

	const maxIter = 8
	dim := len(features[indice[lhsIndex]])
	lhsCenter := make([]float32, dim)
	rhsCenter := make([]float32, dim)
	lhsCount := 1
	rhsCount := 1
	for i := 0; i < dim; i++ {
		lhsCenter[i] = float32(features[indice[lhsIndex]][i])
		rhsCenter[i] = float32(features[indice[rhsIndex]][i])
	}
	nSamples := uint(len(indice))
	if 0 < sampleFeatures && sampleFeatures < nSamples {
		nSamples = sampleFeatures
	}

	for i := 0; i < maxIter; i++ {
		rand.Shuffle(len(indice), func(i, j int) { indice[i], indice[j] = indice[j], indice[i] })
		for _, k := range indice[:nSamples] {
			feature := features[k]
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

	cutPlane := rpCutPlane[T]{
		Normal: normal,
		A:      a,
	}
	return &cutPlane, nil
}

type RpTreeBuilder[T linalg.Number] struct {
	leafs          uint
	sampleFeatures uint
}

func NewRpTreeBuilder[T linalg.Number]() *RpTreeBuilder[T] {
	return &RpTreeBuilder[T]{
		leafs:          rpDefaultLeafs,
		sampleFeatures: rpDefaultSampleFeatures,
	}
}

func (rtb *RpTreeBuilder[T]) SetLeafs(leafs uint) *RpTreeBuilder[T] {
	rtb.leafs = leafs
	return rtb
}

func (rtb *RpTreeBuilder[T]) SetSampleFeatures(sampleFeatures uint) *RpTreeBuilder[T] {
	rtb.sampleFeatures = sampleFeatures
	return rtb
}

func (rtb *RpTreeBuilder[T]) GetPrameterString() string {
	return fmt.Sprintf("leafs=%d_sampleFeatures=%d", rtb.leafs, rtb.sampleFeatures)
}

func (rtb *RpTreeBuilder[T]) Build(features [][]T, env linalg.Env[T]) (BspTree[T], error) {
	indice := make([]int, len(features))
	for i := range indice {
		indice[i] = i
	}
	rand.Shuffle(len(indice), func(i, j int) { indice[i], indice[j] = indice[j], indice[i] })

	bsp_tree := BspTree[T]{
		Indice: indice,
		Nodes:  []Node[T]{},
	}

	cf := func(features [][]T, indice []int, env linalg.Env[T]) (CutPlane[T], error) {
		return newRpCutPlane(features, indice, rtb.sampleFeatures, env)
	}
	_, err := bsp_tree.buildSubTree(features, indice, rtb.leafs, 0, env, cf)
	if err != nil {
		return bsp_tree, err
	}

	return bsp_tree, nil
}
