package collection

type Predicate[T any] interface {
	Evaluate(a T) bool
}

func Partition[T any, P Predicate[T]](buf []T, predicate P) ([]T, []T) {
	i, j := uint(0), uint(len(buf)-1)
	for i <= j {
		for i <= j && !predicate.Evaluate(buf[i]) {
			i++
		}
		for i <= j && predicate.Evaluate(buf[j]) {
			j--
		}
		if i < j {
			buf[i], buf[j] = buf[j], buf[i]
		}
	}
	return buf[:i], buf[i:]
}
