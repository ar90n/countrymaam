package countrymaam

import (
	"context"
	"io"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
)

type SearchResult struct {
	Index    uint
	Distance float32
}

type Index[T linalg.Number] interface {
	SearchChannel(ctx context.Context, query []T) <-chan SearchResult
	Save(reader io.Writer) error
}

type MutableIndex[T linalg.Number] interface {
	SearchChannel(ctx context.Context, query []T) <-chan SearchResult
	Save(reader io.Writer) error
	Add(feature []T)
}

type EntryPointIndex[T linalg.Number] interface {
	SearchChannel(ctx context.Context, query []T) <-chan SearchResult
	SearchChannelWithEntries(ctx context.Context, query []T, entries []uint) <-chan SearchResult
	Save(reader io.Writer) error
}

type IndexBuilder[T linalg.Number, I Index[T]] interface {
	Build(ctx context.Context, features [][]T) (*I, error)
	GetPrameterString() string
}

func Search(ch <-chan SearchResult, n uint, maxCandidates uint) ([]SearchResult, error) {
	items := make([]collection.WithPriority[uint], 0, maxCandidates)
	for item := range ch {
		if maxCandidates <= uint(len(items)) {
			break
		}
		items = append(items, collection.WithPriority[uint]{Item: item.Index, Priority: item.Distance})
	}
	pq := collection.NewPriorityQueueFromSlice(items)

	// take unique neighbors
	ret := make([]SearchResult, 0, n)
	founds := make(map[uint]struct{}, maxCandidates)
	for uint(len(ret)) < n {
		item, err := pq.PopWithPriority()
		if err != nil {
			if err == collection.ErrEmptyPriorityQueue {
				break
			}
			return nil, err
		}

		if _, ok := founds[item.Item]; ok {
			continue
		}
		founds[item.Item] = struct{}{}

		ret = append(ret, SearchResult{Index: item.Item, Distance: item.Priority})
	}

	return ret, nil
}
