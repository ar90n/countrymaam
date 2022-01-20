package number

import (
	"constraints"
)

type Number interface {
	constraints.Integer | constraints.Float
}

func Cast[T Number, U Number](v T) U {
	return U(v)
}

func Abs[T Number](x T) T {
	if x < 0 {
		return -x
	}
	return x
}
func Min[T Number](x, y T) T {
	if x < y {
		return x
	}
	return y
}

func Max[T Number](x, y T) T {
	if x > y {
		return x
	}
	return y
}

func CalcSqDistance[T Number](x, y []T) float64 {
	dist := 0.0
	for i := range x {
		diff := float64(x[i]) - float64(y[i])
		dist += diff * diff
	}

	return dist
}
