package countrymaam

import (
	my_constraints "github.com/ar90n/countrymaam/constraints"
)

type Index[T my_constraints.Number, U any] interface {
	Add(feature []T, item U)
	Search(feature []T, n uint, r float32) []U
	Build() error
}
