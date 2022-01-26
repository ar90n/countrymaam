package countrymaam

import (
	"bytes"
	"encoding/gob"
	"io"

	"github.com/ar90n/countrymaam/number"
)

type Index[T number.Number, U any] interface {
	Add(feature []T, item U)
	Search(feature []T, n uint, maxCandidates uint) ([]Candidate[U], error)
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

func NewKdTreeIndex[T number.Number, U any](dim uint, leafSize uint) *bspTreeIndex[T, U, kdCutPlane[T, U]] {
	gob.Register(kdCutPlane[T, U]{})
	return &bspTreeIndex[T, U, kdCutPlane[T, U]]{
		Dim:      dim,
		Pool:     make([]treeElement[T, U], 0, 4096),
		Roots:    make([]*treeNode[T, U], 1),
		LeafSize: leafSize,
	}
}

func LoadKdTreeIndex[T number.Number, U any](r io.Reader) (*bspTreeIndex[T, U, kdCutPlane[T, U]], error) {
	gob.Register(kdCutPlane[T, U]{})
	index, err := loadIndex[bspTreeIndex[T, U, kdCutPlane[T, U]]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}

func NewRpTreeIndex[T number.Number, U any](dim uint, leafSize uint) *bspTreeIndex[T, U, rpCutPlane[T, U]] {
	gob.Register(rpCutPlane[T, U]{})
	return &bspTreeIndex[T, U, rpCutPlane[T, U]]{
		Dim:      dim,
		Pool:     make([]treeElement[T, U], 0),
		Roots:    make([]*treeNode[T, U], 1),
		LeafSize: leafSize,
	}
}

func LoadRpTreeIndex[T number.Number, U any](r io.Reader) (*bspTreeIndex[T, U, rpCutPlane[T, U]], error) {
	gob.Register(rpCutPlane[T, U]{})
	index, err := loadIndex[bspTreeIndex[T, U, rpCutPlane[T, U]]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}

func NewRandomizedKdTreeIndex[T number.Number, U any](dim uint, leafSize uint, nTrees uint) *bspTreeIndex[T, U, randomizedKdCutPlane[T, U]] {
	gob.Register(kdCutPlane[T, U]{})
	gob.Register(randomizedKdCutPlane[T, U]{})
	return &bspTreeIndex[T, U, randomizedKdCutPlane[T, U]]{
		Dim:      dim,
		Pool:     make([]treeElement[T, U], 0),
		Roots:    make([]*treeNode[T, U], nTrees),
		LeafSize: leafSize,
	}
}

func LoadRandomizedKdTreeIndex[T number.Number, U any](r io.Reader) (*bspTreeIndex[T, U, randomizedKdCutPlane[T, U]], error) {
	gob.Register(kdCutPlane[T, U]{})
	gob.Register(randomizedKdCutPlane[T, U]{})
	index, err := loadIndex[bspTreeIndex[T, U, randomizedKdCutPlane[T, U]]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}

func NewRandomizedRpTreeIndex[T number.Number, U any](dim uint, leafSize uint, nTrees uint) *bspTreeIndex[T, U, rpCutPlane[T, U]] {
	gob.Register(rpCutPlane[T, U]{})
	return &bspTreeIndex[T, U, rpCutPlane[T, U]]{
		Dim:      dim,
		Pool:     make([]treeElement[T, U], 0, 4096),
		Roots:    make([]*treeNode[T, U], nTrees),
		LeafSize: leafSize,
	}
}

func LoadRandomizedRpTreeIndex[T number.Number, U any](r io.Reader) (*bspTreeIndex[T, U, rpCutPlane[T, U]], error) {
	gob.Register(rpCutPlane[T, U]{})
	index, err := loadIndex[bspTreeIndex[T, U, rpCutPlane[T, U]]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}
