package countrymaam

import (
	"github.com/ar90n/countrymaam/number"
)

type Index[T number.Number, U any] interface {
	Add(feature []T, item U)
	Search(feature []T, n uint, r float64) ([]Candidate[U], error)
	Build() error
}

type Candidate[U any] struct {
	Distance float64
	Item     U
}

func NewFlatIndex[T number.Number, U any](dim uint) *flatIndex[T, U] {
	return &flatIndex[T, U]{
		dim:      dim,
		features: make([][]T, 0),
		items:    make([]U, 0),
	}
}

func NewKdTreeIndex[T number.Number, U any](dim uint, leafSize uint) *bspTreeIndex[T, U] {
	return &bspTreeIndex[T, U]{
		dim:         dim,
		pool:        make([]*treeElement[T, U], 0),
		roots:       make([]*treeNode[T, U], 1),
		leafSize:    leafSize,
		newCutPlane: NewKdCutPlane[T, *treeElement[T, U]],
	}
}

func NewRpTreeIndex[T number.Number, U any](dim uint, leafSize uint) *bspTreeIndex[T, U] {
	return &bspTreeIndex[T, U]{
		dim:         dim,
		pool:        make([]*treeElement[T, U], 0),
		roots:       make([]*treeNode[T, U], 1),
		leafSize:    leafSize,
		newCutPlane: NewRpCutPlane[T, *treeElement[T, U]],
	}
}

func NewRandomizedKdTreeIndex[T number.Number, U any](dim uint, leafSize uint, nTrees uint) *bspTreeIndex[T, U] {
	return &bspTreeIndex[T, U]{
		dim:         dim,
		pool:        make([]*treeElement[T, U], 0),
		roots:       make([]*treeNode[T, U], nTrees),
		leafSize:    leafSize,
		newCutPlane: NewRandomizedKdCutPlane[T, *treeElement[T, U]],
	}
}

func NewRandomizedRpTreeIndex[T number.Number, U any](dim uint, leafSize uint, nTrees uint) *bspTreeIndex[T, U] {
	return &bspTreeIndex[T, U]{
		dim:         dim,
		pool:        make([]*treeElement[T, U], 0),
		roots:       make([]*treeNode[T, U], nTrees),
		leafSize:    leafSize,
		newCutPlane: NewRpCutPlane[T, *treeElement[T, U]],
	}
}
