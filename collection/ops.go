package collection

func Partition[T any](buf []T, predicate func(T) bool) uint {
	i, j := uint(0), uint(len(buf)-1)
	for i <= j {
		for i <= j && !predicate(buf[i]) {
			i++
		}
		for i <= j && predicate(buf[j]) {
			j--
		}
		if i < j {
			buf[i], buf[j] = buf[j], buf[i]
		}
	}
	return i
}
