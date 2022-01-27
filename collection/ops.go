package collection

func Partition[T any](buf []T, predicate func(T) bool) uint {
	if len(buf) == 0 {
		return 0
	}

	i, j := 0, len(buf)-1
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
	return uint(i)
}
