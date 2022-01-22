package countrymaam

import (
	"io"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/number"
)

type flatIndex[T number.Number, U any] struct {
	Dim      uint
	Features [][]T
	Items    []U
}

var _ = (*flatIndex[float32, int])(nil)

func (fi *flatIndex[T, U]) Add(feature []T, item U) {
	fi.Features = append(fi.Features, feature)
	fi.Items = append(fi.Items, item)
}

func (fi flatIndex[T, U]) Search(query []T, n uint, r float64) ([]Candidate[U], error) {
	candidates := collection.NewPriorityQueue[U](int(n))

	for i, feature := range fi.Features {
		distance := number.CalcSqDistance(query, feature)
		if distance < r {
			candidates.Push(fi.Items[i], distance)
		}
	}

	items := make([]Candidate[U], number.Min(n, uint(candidates.Len())))
	for i := range items {
		item, err := candidates.PopWithPriority()
		if err != nil {
			return nil, err
		}

		items[i].Item = item.Item
		items[i].Distance = item.Priority
	}

	return items, nil
}

func (fi flatIndex[T, U]) Build() error {
	return nil
}

func (fi flatIndex[T, U]) HasIndex() bool {
	return true
}

func (fi flatIndex[T, U]) Save(w io.Writer) error {
	return saveIndex(&fi, w)
}
