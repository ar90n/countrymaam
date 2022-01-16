package constraints

import (
	"constraints"
)

type Number interface {
	constraints.Integer | constraints.Float
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
