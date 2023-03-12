package countrymaam

import (
	"context"
	"io"

	"github.com/ar90n/countrymaam/linalg"
)

type Candidate[U comparable] struct {
	Distance float64
	Item     U
}

type Index[T linalg.Number, U comparable] interface {
	Search(ctx context.Context, feature []T, n uint, maxCandidates uint) ([]Candidate[U], error)
	SearchChannel(ctx context.Context, feature []T) <-chan Candidate[U]
	Save(reader io.Writer) error
}

type IndexBuilder[T linalg.Number, U comparable] interface {
	Build(ctx context.Context, features [][]T, items []U) (Index[T, U], error)
}

type MutableIndex[T linalg.Number, U comparable] interface {
	Index[T, U]
	Add(feature []T, item U)
}

type LegacyIndex[T linalg.Number, U comparable] interface {
	Search(ctx context.Context, feature []T, n uint, maxCandidates uint) ([]Candidate[U], error)
	Add(feature []T, item U)
	SearchChannel(ctx context.Context, feature []T) <-chan Candidate[U]
	Build(ctx context.Context) error
	HasIndex() bool
	Save(reader io.Writer) error
}
