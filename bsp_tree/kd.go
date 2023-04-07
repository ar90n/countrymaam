package bsp_tree

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
)

const kdTreeDefaultLeafs = 16

var (
	_ CutPlane[float32] = (*kdCutPlane[float32])(nil)
)

// kdCutPlane is a cut plane that is constructed by kdtree algorithm.
// this is derived from flann library.
// https://github.com/flann-lib/flann/blob/master/src/cpp/flann/algorithms/kdtree_index.h
type kdCutPlane[T linalg.Number] struct {
	Axis  uint
	Value float64
}

func (cp kdCutPlane[T]) Evaluate(feature []T, env linalg.Env[T]) bool {
	return 0.0 <= cp.Distance(feature, env)
}

func (cp kdCutPlane[T]) Distance(feature []T, env linalg.Env[T]) float64 {
	return float64(feature[cp.Axis]) - cp.Value
}

func newKdCutPlane[T linalg.Number](features [][]T, indice []int, nFeatures uint, nCandidates int, env linalg.Env[T]) (CutPlane[T], error) {
	if len(indice) == 0 {
		return nil, errors.New("elements is empty")
	}

	dim := len(features[0])
	accs := make([]float64, dim)
	sqAccs := make([]float64, dim)
	nSamples := uint(len(indice))
	if 0 < nFeatures && nFeatures < nSamples {
		nSamples = nFeatures
	}
	for _, i := range indice[:nSamples] {
		for j, v := range features[i] {
			v := float64(v)
			accs[j] += v
			sqAccs[j] += v * v
		}
	}

	invN := 1.0 / float64(nSamples)
	queue := collection.PriorityQueue[*kdCutPlane[T]]{}
	for i := range accs {
		mean := accs[i] * invN
		sqMean := sqAccs[i] * invN
		variance := sqMean - mean*mean

		cutPlane := &kdCutPlane[T]{
			Axis:  uint(i),
			Value: mean,
		}
		queue.Push(cutPlane, -float32(variance))
	}

	// Randomly select one of the best candidates.
	nCandidates = linalg.Min(int(nCandidates), queue.Len())
	nSkip := 0
	if 0 < nCandidates {
		nSkip = rand.Intn(nCandidates) - 1
	}
	for i := 0; i < nSkip; i++ {
		_, err := queue.Pop()
		if err != nil {
			return nil, err
		}
	}
	return queue.Pop()
}

type KdTreeBuilder[T linalg.Number] struct {
	leafs          uint
	sampleFeatures uint
	topKCandidates uint
}

func NewKdTreeBuilder[T linalg.Number]() *KdTreeBuilder[T] {
	return &KdTreeBuilder[T]{
		leafs: kdTreeDefaultLeafs,
	}
}

func (ktb *KdTreeBuilder[T]) SetLeafs(leafs uint) *KdTreeBuilder[T] {
	ktb.leafs = leafs
	return ktb
}

func (ktb *KdTreeBuilder[T]) SetSampleFeatures(sampleFeatures uint) *KdTreeBuilder[T] {
	ktb.sampleFeatures = sampleFeatures
	return ktb
}

func (ktb *KdTreeBuilder[T]) SetTopKCandidates(topKCandidates uint) *KdTreeBuilder[T] {
	ktb.topKCandidates = topKCandidates
	return ktb
}

func (ktb KdTreeBuilder[T]) GetPrameterString() string {
	return fmt.Sprintf("leafs=%d_sampleFeatures=%d_topKCandidates=%d", ktb.leafs, ktb.sampleFeatures, ktb.topKCandidates)
}

func (ktb *KdTreeBuilder[T]) Build(features [][]T, env linalg.Env[T]) (BspTree[T], error) {
	//gob.Register(kdCutPlane[T]{})

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
		return newKdCutPlane(features, indice, ktb.sampleFeatures, int(ktb.topKCandidates), env)
	}
	_, err := bsp_tree.buildSubTree(features, indice, ktb.leafs, 0, env, cf)
	if err != nil {
		return bsp_tree, err
	}

	return bsp_tree, nil
}
