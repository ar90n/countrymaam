package countrymaam

import (
	"context"
	"io"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
)

type Candidate[U comparable] struct {
	Distance float32
	Item     U
}

type Index[T linalg.Number, U comparable] interface {
	SearchChannel(ctx context.Context, query []T) <-chan Candidate[U]
	Save(reader io.Writer) error
}

type MutableIndex[T linalg.Number, U comparable] interface {
	SearchChannel(ctx context.Context, query []T) <-chan Candidate[U]
	Save(reader io.Writer) error
	Add(feature []T, item U)
}

type EntryPointIndex[T linalg.Number, U comparable] interface {
	SearchChannel(ctx context.Context, query []T) <-chan Candidate[U]
	SearchChannelWithEntries(ctx context.Context, query []T, entries []uint) <-chan Candidate[U]
	Save(reader io.Writer) error
}

type IndexBuilder[T linalg.Number, U comparable] interface {
	Build(ctx context.Context, features [][]T, items []U) (Index[T, U], error)
}

func Search[U comparable](ch <-chan Candidate[U], n uint, maxCandidates uint) ([]Candidate[U], error) {
	items := make([]collection.WithPriority[U], 0, maxCandidates)
	for item := range ch {
		if maxCandidates <= uint(len(items)) {
			break
		}
		items = append(items, collection.WithPriority[U]{Item: item.Item, Priority: item.Distance})
	}
	pq := collection.NewPriorityQueueFromSlice(items)

	// take unique neighbors
	ret := make([]Candidate[U], 0, n)
	founds := make(map[U]struct{}, maxCandidates)
	for uint(len(ret)) < n {
		item, err := pq.PopWithPriority()
		if err != nil {
			return nil, err
		}

		if _, ok := founds[item.Item]; ok {
			continue
		}
		founds[item.Item] = struct{}{}

		ret = append(ret, Candidate[U]{Item: item.Item, Distance: item.Priority})
	}
	return ret, nil
}
