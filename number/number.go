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

func CalcSqDist[T Number, U Number](x []T, y []U) float64 {
	dist := 0.0

	i := 0
	for ; i < len(x)%4; i++ {
		diff := float64(x[i]) - float64(y[i])
		dist += diff * diff
	}

	for ; i < len(x); i += 4 {
		diff0 := float64(x[i+0]) - float64(y[i+0])
		diff1 := float64(x[i+1]) - float64(y[i+1])
		diff2 := float64(x[i+2]) - float64(y[i+2])
		diff3 := float64(x[i+3]) - float64(y[i+3])
		dist += diff0*diff0 + diff1*diff1 + diff2*diff2 + diff3*diff3
	}

	return dist
}

func CalcDot[T Number, U Number](x []T, y []U) float64 {
	dot := 0.0

	i := 0
	for ; i < len(x)%4; i++ {
		dot += float64(x[i]) * float64(y[i])
	}

	for ; i < len(x); i += 4 {
		mul0 := float64(x[i+0]) * float64(y[i+0])
		mul1 := float64(x[i+1]) * float64(y[i+1])
		mul2 := float64(x[i+2]) * float64(y[i+2])
		mul3 := float64(x[i+3]) * float64(y[i+3])
		dot += mul0 + mul1 + mul2 + mul3
	}

	return dot
}
