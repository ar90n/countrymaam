package metric

import (
	"github.com/ar90n/countrymaam/number"
)

type Metric[T number.Number] interface {
	CalcDistance(a []T, b []T) float32
}

type SqL2Dist[T number.Number] struct {
}

func (em SqL2Dist[T]) CalcDistance(lhs, rhs []T) float32 {
	var sum float32
	for i := range lhs {
		diff := float32(lhs[i] - rhs[i])
		sum += diff * diff
	}
	return sum
}
