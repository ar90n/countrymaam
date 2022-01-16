package metric

import (
	my_constraints "github.com/ar90n/countrymaam/constraints"
)

type Metric[T my_constraints.Number] interface {
	CalcDistance(a []T, b []T) float32
}

type SqL2Dist[T my_constraints.Number] struct {
}

func (em SqL2Dist[T]) CalcDistance(lhs, rhs []T) float32 {
	var sum float32
	for i := range lhs {
		diff := float32(lhs[i] - rhs[i])
		sum += diff * diff
	}
	return sum
}
