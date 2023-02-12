package countrymaam

import (
	"context"
	"io"

	"github.com/ar90n/countrymaam/index"
	"github.com/ar90n/countrymaam/linalg"
)

type Index[T linalg.Number, U comparable] interface {
	Add(feature []T, item U)
	Search(ctx context.Context, feature []T, n uint, maxCandidates uint) ([]index.Candidate[U], error)
	SearchChannel(ctx context.Context, feature []T) <-chan index.Candidate[U]
	Build(ctx context.Context) error
	HasIndex() bool
	Save(reader io.Writer) error
}
