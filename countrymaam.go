package countrymaam

import (
	"github.com/ar90n/countrymaam/index"
	"github.com/ar90n/countrymaam/number"
)

type Index[T number.Number, U any] interface {
	Add(feature []T, item U)
	Search(feature []T, n uint, r float64) ([]index.Candidate[U], error)
	Build() error
}
