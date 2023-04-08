package index

import (
	"context"
	"io"
	"runtime"
	"sync"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
)

type FlatIndex[T linalg.Number] struct {
	Features      [][]T
	MaxGoroutines uint
}

var _ = (*FlatIndex[float32])(nil)

type chunk struct {
	Begin uint
	End   uint
}

func (fi FlatIndex[T]) SearchChannel(ctx context.Context, query []T) <-chan countrymaam.SearchResult {
	featStream := make(chan collection.WithPriority[uint])
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
					case featStream <- collection.WithPriority[uint]{
						Item:     i,
						Priority: distance,
					}:
					}
				}
			}(c)
		}

		wg.Wait()
	}()

	outputStream := make(chan countrymaam.SearchResult)
	go func() {
		defer close(outputStream)

		candidates := collection.NewPriorityQueue[uint](32)
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
			case outputStream <- countrymaam.SearchResult{
				Index:    item.Item,
				Distance: item.Priority,
			}:
			}
		}
	}()

	return outputStream
}

func (fi FlatIndex[T]) Save(w io.Writer) error {
	return saveIndex(fi, w)
}

func (fi *FlatIndex[T]) Add(feature []T) {
	fi.Features = append(fi.Features, feature)
}

func (fi FlatIndex[T]) getChunks(procs uint) <-chan chunk {
	ch := make(chan chunk)
	go func() {
		defer close(ch)

		n := procs
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

type FlatIndexBuilder[T linalg.Number] struct {
	dim           uint
	maxGoroutines int
}

func NewFlatIndexBuilder[T linalg.Number](dim uint) *FlatIndexBuilder[T] {
	return &FlatIndexBuilder[T]{
		dim:           dim,
		maxGoroutines: int(runtime.NumCPU()),
	}
}

func (fig FlatIndexBuilder[T]) Build(ctx context.Context, features [][]T) (*FlatIndex[T], error) {
	if err := fig.validate(features); err != nil {
		return nil, err
	}

	index := &FlatIndex[T]{
		Features:      features,
		MaxGoroutines: uint(fig.maxGoroutines),
	}
	return index, nil
}

func (fig *FlatIndexBuilder[T]) SetMaxGoroutines(maxGoroutines uint) {
	fig.maxGoroutines = int(maxGoroutines)
}

func (fig FlatIndexBuilder[T]) GetPrameterString() string {
	return ""
}

func (fig FlatIndexBuilder[T]) validate(features [][]T) error {
	for _, feature := range features {
		if uint(len(feature)) != fig.dim {
			return countrymaam.ErrInvalidFeatureDim
		}
	}

	return nil
}

func LoadFlatIndex[T linalg.Number](r io.Reader) (*FlatIndex[T], error) {
	index, err := loadIndex[FlatIndex[T]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}
