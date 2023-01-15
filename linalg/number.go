package linalg

type Number interface {
	~uint8 | ~float32 | ~float64 | ~uint | ~int
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
