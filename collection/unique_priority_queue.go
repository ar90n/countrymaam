package collection

type UniquePriorityQueue[T comparable] struct {
	PriorityQueue[T]
	set map[T]struct{}
}

func (upq *UniquePriorityQueue[T]) Push(item T, priority float64) {
	if _, ok := upq.set[item]; ok {
		return
	}
	if upq.set == nil {
		upq.set = make(map[T]struct{})
	}
	upq.set[item] = struct{}{}

	upq.PriorityQueue.Push(item, priority)
}

func (upq *UniquePriorityQueue[T]) Pop() (T, error) {
	item, err := upq.PriorityQueue.Pop()
	if err != nil {
		return item, err
	}
	delete(upq.set, item)

	return item, nil
}
