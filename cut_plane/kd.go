package cut_plane

import (
	"errors"
	"math/rand"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/index"
	"github.com/ar90n/countrymaam/linalg"
)

// KdCutPlane is a cut plane that is constructed by kdtree algorithm.
// this is derived from flann library.
// https://github.com/flann-lib/flann/blob/master/src/cpp/flann/algorithms/kdtree_index.h
type KdCutPlane[T linalg.Number, U comparable] struct {
	axis  uint
	value float64
}

func (cp KdCutPlane[T, U]) Evaluate(feature []T, env linalg.Env[T]) bool {
	return 0.0 <= cp.Distance(feature, env)
}

func (cp KdCutPlane[T, U]) Distance(feature []T, env linalg.Env[T]) float64 {
	return float64(feature[cp.axis]) - cp.value
}

type KdCutPlaneFactory[T linalg.Number, U comparable] struct {
	features   uint
	candidates uint
}

func NewKdCutPlaneFactory[T linalg.Number, U comparable](features, candidates uint) index.CutPlaneFactory[T, U] {
	return KdCutPlaneFactory[T, U]{
		features:   features,
		candidates: candidates,
	}
}

func (f KdCutPlaneFactory[T, U]) Default() index.CutPlane[T, U] {
	return KdCutPlane[T, U]{}
}

func (f KdCutPlaneFactory[T, U]) Build(elements []index.TreeElement[T, U], indice []int, env linalg.Env[T]) (index.CutPlane[T, U], error) {
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
	queue := collection.PriorityQueue[*KdCutPlane[T, U]]{}
	for i := range accs {
		mean := accs[i] * invN
		sqMean := sqAccs[i] * invN
		variance := sqMean - mean*mean

		cutPlane := &KdCutPlane[T, U]{
			axis:  uint(i),
			value: mean,
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
