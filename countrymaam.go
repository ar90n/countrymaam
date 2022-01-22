package countrymaam

import (
	"bytes"
	"encoding/gob"
	"io"

	"github.com/ar90n/countrymaam/number"
)

type Index[T number.Number, U any] interface {
	Add(feature []T, item U)
	Search(feature []T, n uint, r float64) ([]Candidate[U], error)
	Build() error
	HasIndex() bool
	Save(reader io.Writer) error
}

type Candidate[U any] struct {
	Distance float64
	Item     U
}

func saveIndex[T any](index *T, w io.Writer) error {
	var buffer bytes.Buffer
	enc := gob.NewEncoder(&buffer)
	if err := enc.Encode(index); err != nil {
		return err
	}

	beg := 0
	byteArray := buffer.Bytes()
	for beg < len(byteArray) {
		n, err := w.Write(byteArray[beg:])
		if err != nil {
			return err
		}
		beg += n
	}

	return nil
}

func loadIndex[T any](r io.Reader) (ret T, _ error) {
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&ret); err != nil {
		return ret, err
	}

	return ret, nil
}

func NewFlatIndex[T number.Number, U any](dim uint) *flatIndex[T, U] {
	return &flatIndex[T, U]{
		Dim:      dim,
		Features: make([][]T, 0),
		Items:    make([]U, 0),
	}
}

func LoadFlatIndex[T number.Number, U any](r io.Reader) (*flatIndex[T, U], error) {
	index, err := loadIndex[flatIndex[T, U]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}

func NewKdTreeIndex[T number.Number, U any](dim uint, leafSize uint) *bspTreeIndex[T, U, kdCutPlane[T]] {
	gob.Register(kdCutPlane[T]{})
	return &bspTreeIndex[T, U, kdCutPlane[T]]{
		Dim:      dim,
		Pool:     make([]*treeElement[T, U], 0),
		Roots:    make([]*treeNode[T, U], 1),
		LeafSize: leafSize,
	}
}

func LoadKdTreeIndex[T number.Number, U any](r io.Reader) (*bspTreeIndex[T, U, kdCutPlane[T]], error) {
	gob.Register(kdCutPlane[T]{})
	index, err := loadIndex[bspTreeIndex[T, U, kdCutPlane[T]]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}

func NewRpTreeIndex[T number.Number, U any](dim uint, leafSize uint) *bspTreeIndex[T, U, rpCutPlane[T]] {
	gob.Register(rpCutPlane[T]{})
	return &bspTreeIndex[T, U, rpCutPlane[T]]{
		Dim:      dim,
		Pool:     make([]*treeElement[T, U], 0),
		Roots:    make([]*treeNode[T, U], 1),
		LeafSize: leafSize,
	}
}

func LoadRpTreeIndex[T number.Number, U any](r io.Reader) (*bspTreeIndex[T, U, rpCutPlane[T]], error) {
	gob.Register(rpCutPlane[T]{})
	index, err := loadIndex[bspTreeIndex[T, U, rpCutPlane[T]]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}

func NewRandomizedKdTreeIndex[T number.Number, U any](dim uint, leafSize uint, nTrees uint) *bspTreeIndex[T, U, randomizedKdCutPlane[T]] {
	gob.Register(randomizedKdCutPlane[T]{})
	return &bspTreeIndex[T, U, randomizedKdCutPlane[T]]{
		Dim:      dim,
		Pool:     make([]*treeElement[T, U], 0),
		Roots:    make([]*treeNode[T, U], nTrees),
		LeafSize: leafSize,
	}
}

func LoadRandomizedKdTreeIndex[T number.Number, U any](r io.Reader) (*bspTreeIndex[T, U, randomizedKdCutPlane[T]], error) {
	gob.Register(randomizedKdCutPlane[T]{})
	index, err := loadIndex[bspTreeIndex[T, U, randomizedKdCutPlane[T]]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}

func NewRandomizedRpTreeIndex[T number.Number, U any](dim uint, leafSize uint, nTrees uint) *bspTreeIndex[T, U, rpCutPlane[T]] {
	gob.Register(rpCutPlane[T]{})
	return &bspTreeIndex[T, U, rpCutPlane[T]]{
		Dim:      dim,
		Pool:     make([]*treeElement[T, U], 0),
		Roots:    make([]*treeNode[T, U], nTrees),
		LeafSize: leafSize,
	}
}

func LoadRandomizedRpTreeIndex[T number.Number, U any](r io.Reader) (*bspTreeIndex[T, U, rpCutPlane[T]], error) {
	gob.Register(rpCutPlane[T]{})
	index, err := loadIndex[bspTreeIndex[T, U, rpCutPlane[T]]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}
