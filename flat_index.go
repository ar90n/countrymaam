package countrymaam

import (
	"context"
	"io"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
)

type flatIndex[T linalg.Number, U any] struct {
	Dim           uint
	Features      [][]T
	Items         []U
	env           linalg.Env[T]
	maxCandidates uint
}

var _ = (*flatIndex[float32, int])(nil)

func (fi *flatIndex[T, U]) Add(feature []T, item U) {
	fi.Features = append(fi.Features, feature)
	fi.Items = append(fi.Items, item)
}

func (fi flatIndex[T, U]) Search(query []T, n uint, maxCandidates uint) ([]Candidate[U], error) {
	candidates := collection.NewPriorityQueue[U](int(n))

	for i, feature := range fi.Features {
		distance := fi.env.SqL2(query, feature)
		candidates.Push(fi.Items[i], float64(distance))
	}

	items := make([]Candidate[U], linalg.Min(n, uint(candidates.Len())))
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

func (fi flatIndex[T, U]) Search2(ctx context.Context, query []T) <-chan Candidate[U] {
	ch := make(chan Candidate[U])
	go func() {
		defer close(ch)

		candidates := collection.NewPriorityQueue[U](int(fi.maxCandidates))

		for i, feature := range fi.Features {
			distance := fi.env.SqL2(query, feature)
			candidates.Push(fi.Items[i], float64(distance))
		}

		for item := range candidates.PopWithPriority2() {
			select {
			case <-ctx.Done():
				return
			case ch <- Candidate[U]{
				Item:     item.Item,
				Distance: item.Priority,
			}:
			}
		}
	}()

	return ch
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
