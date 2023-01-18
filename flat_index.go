package countrymaam

import (
	"context"
	"io"
	"runtime"
	"sort"
	"sync"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/ar90n/countrymaam/pipeline"
)

type flatIndex[T linalg.Number, U comparable] struct {
	Dim           uint
	Features      [][]T
	Items         []U
	env           linalg.Env[T]
	maxCandidates uint
	nProc         uint
}

var _ = (*flatIndex[float32, int])(nil)

type chunk struct {
	Begin uint
	End   uint
}

func (fi *flatIndex[T, U]) Add(feature []T, item U) {
	fi.Features = append(fi.Features, feature)
	fi.Items = append(fi.Items, item)
}

func (fi flatIndex[T, U]) Search(ctx context.Context, query []T, n uint, maxCandidates uint) ([]Candidate[U], error) {
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
	return items, nil
}

func (fi flatIndex[T, U]) SearchChannel(ctx context.Context, query []T) <-chan Candidate[U] {
	featStream := make(chan collection.WithPriority[U])
	go func() {
		defer close(featStream)

		wg := sync.WaitGroup{}
		for c := range fi.getChunks() {
			wg.Add(1)
			go func(c chunk) {
				defer wg.Done()

				for i := c.Begin; i < c.End; i++ {
					distance := fi.env.SqL2(query, fi.Features[i])
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

	outputStream := make(chan Candidate[U])
	go func() {
		defer close(outputStream)

		candidates := collection.NewPriorityQueue[U](int(fi.maxCandidates))
		for item := range featStream {
			candidates.Push(item.Item, item.Priority)
		}

		for item := range candidates.PopWithPriority2(ctx) {
			select {
			case <-ctx.Done():
				return
			case outputStream <- Candidate[U]{
				Item:     item.Item,
				Distance: float64(item.Priority),
			}:
			}
		}
	}()

	return outputStream
}

func (fi flatIndex[T, U]) Build(ctx context.Context) error {
	return nil
}

func (fi flatIndex[T, U]) HasIndex() bool {
	return true
}

func (fi flatIndex[T, U]) Save(w io.Writer) error {
	return saveIndex(&fi, w)
}

func (fi flatIndex[T, U]) getChunks() <-chan chunk {
	ch := make(chan chunk)
	go func() {
		defer close(ch)

		n := GetProcNum(fi.nProc)
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

func GetProcNum(maxGoRoutines uint) uint {
	if maxGoRoutines == 0 {
		return uint(runtime.NumCPU())
	}

	return maxGoRoutines
}
