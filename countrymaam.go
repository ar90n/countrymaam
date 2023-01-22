package countrymaam

import (
	"context"
	"io"

	"github.com/ar90n/countrymaam/linalg"
)

type Index[T linalg.Number, U comparable] interface {
	Add(feature []T, item U)
	Search(ctx context.Context, feature []T, n uint, maxCandidates uint) ([]Candidate[U], error)
	SearchChannel(ctx context.Context, feature []T) <-chan Candidate[U]
	Build(ctx context.Context) error
	HasIndex() bool
	Save(reader io.Writer) error
}

type Candidate[U comparable] struct {
	Distance float64
	Item     U
}
