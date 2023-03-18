package index

import (
	"context"
	"io"
	"sort"
	"sync"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/common"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/ar90n/countrymaam/pipeline"
)

type FlatIndex[T linalg.Number, U comparable] struct {
	Features      [][]T
	Items         []U
	MaxGoroutines uint
}

var _ = (*FlatIndex[float32, int])(nil)

type chunk struct {
	Begin uint
	End   uint
}

func (fi FlatIndex[T, U]) Search(ctx context.Context, query []T, n uint, maxCandidates uint) ([]countrymaam.Candidate[U], error) {
	ch := fi.SearchChannel(ctx, query)
	ch = pipeline.Unique(ctx, ch)
	ch = pipeline.Take(ctx, maxCandidates, ch)
	items := pipeline.ToSlice(ctx, ch)
	sort.Slice(items, func(i, j int) bool {
		return items[i].Distance < items[j].Distance
	})

	if uint(len(items)) < n {
		n = uint(len(items))
	}
	return items[:n], nil
}

func (fi FlatIndex[T, U]) SearchChannel(ctx context.Context, query []T) <-chan countrymaam.Candidate[U] {
	featStream := make(chan collection.WithPriority[U])
	go func() {
		defer close(featStream)

		env := linalg.NewLinAlgFromContext[T](ctx)

		wg := sync.WaitGroup{}
		for c := range fi.getChunks(fi.MaxGoroutines) {
			wg.Add(1)
			go func(c chunk) {
				defer wg.Done()

				for i := c.Begin; i < c.End; i++ {
					distance := env.SqL2(query, fi.Features[i])
					select {
					case <-ctx.Done():
						return
					case featStream <- collection.WithPriority[U]{
						Item:     fi.Items[i],
						Priority: distance,
					}:
					}
				}
			}(c)
		}

		wg.Wait()
	}()

	outputStream := make(chan countrymaam.Candidate[U])
	go func() {
		defer close(outputStream)

		candidates := collection.NewPriorityQueue[U](32)
		for item := range featStream {
			candidates.Push(item.Item, item.Priority)
		}

		for 0 < candidates.Len() {
			item, err := candidates.PopWithPriority()
			if err != nil {
				return
			}
			select {
			case <-ctx.Done():
				return
			case outputStream <- countrymaam.Candidate[U]{
				Item:     item.Item,
				Distance: item.Priority,
			}:
			}
		}
	}()

	return outputStream
}

func (fi FlatIndex[T, U]) Save(w io.Writer) error {
	return saveIndex(fi, w)
}

func (fi *FlatIndex[T, U]) Add(feature []T, item U) {
	fi.Features = append(fi.Features, feature)
	fi.Items = append(fi.Items, item)
}

func (fi FlatIndex[T, U]) getChunks(procs uint) <-chan chunk {
	ch := make(chan chunk)
	go func() {
		defer close(ch)

		n := common.GetProcNum(procs)
		bs := uint(len(fi.Features)) / n
		rem := uint(len(fi.Features)) % n
		bi := uint(0)
		for i := uint(0); i < n; i++ {
			ei := bi + bs
			if i < rem {
				ei += 1
			}

			ch <- chunk{Begin: uint(bi), End: uint(ei)}
			bi = ei
		}
	}()

	return ch
}

type FlatIndexBuilder[T linalg.Number, U comparable] struct {
	dim           uint
	maxGoroutines int
}

func NewFlatIndexBuilder[T linalg.Number, U comparable](dim uint) *FlatIndexBuilder[T, U] {
	return &FlatIndexBuilder[T, U]{
		dim: dim,
	}
}

func (fig FlatIndexBuilder[T, U]) Build(ctx context.Context, features [][]T, items []U) (countrymaam.Index[T, U], error) {
	if err := fig.validate(features, items); err != nil {
		return nil, err
	}

	index := &FlatIndex[T, U]{
		Features:      features,
		Items:         items,
		MaxGoroutines: uint(fig.maxGoroutines),
	}
	return index, nil
}

func (fig *FlatIndexBuilder[T, U]) SetMaxGoroutines(maxGoroutines uint) {
	fig.maxGoroutines = int(maxGoroutines)
}

func (fig FlatIndexBuilder[T, U]) validate(features [][]T, items []U) error {
	if uint(len(features)) != uint(len(items)) {
		return countrymaam.ErrInvalidFeaturesAndItems
	}

	for _, feature := range features {
		if uint(len(feature)) != fig.dim {
			return countrymaam.ErrInvalidFeatureDim
		}
	}

	return nil
}

func LoadFlatIndex[T linalg.Number, U comparable](r io.Reader) (*FlatIndex[T, U], error) {
	index, err := loadIndex[FlatIndex[T, U]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}
