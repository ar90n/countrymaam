package collection

import (
	"errors"
	"fmt"
)

type WithPriority[T any] struct {
	Item     T
	Priority float32
}

type priorityQueue[T any] []WithPriority[T]

func (pq priorityQueue[T]) Len() int { return len(pq) }

func (pq *priorityQueue[T]) Push(x WithPriority[T]) {
	*pq = append(*pq, x)
	pq.up(pq.Len() - 1)
}

func (pq *priorityQueue[T]) Pop() WithPriority[T] {
	n := pq.Len() - 1
	pq.swap(0, n)
	pq.down(0, n)

	item := (*pq)[n]
	*pq = (*pq)[0:n]
	return item
}

// derived from container/heap
func (pq priorityQueue[T]) Init() {
	// heapify
	n := pq.Len()
	for i := n/2 - 1; i >= 0; i-- {
		pq.down(i, n)
	}
}

func (pq priorityQueue[T]) less(i, j int) bool {
	return pq[i].Priority < pq[j].Priority
}

func (pq priorityQueue[T]) swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

// derived from container/heap
func (pq priorityQueue[T]) up(j int) {
	for {
		i := (j - 1) / 2 // parent
		if i == j || !pq.less(j, i) {
			break
		}
		pq.swap(i, j)
		j = i
	}
}

// derived from container/heap
func (pq priorityQueue[T]) down(i0, n int) bool {
	i := i0
	for {
		j1 := 2*i + 1
		if n <= j1 || j1 < 0 { // j1 < 0 after int overflow
			break
		}
		j := pq.getLesserSibling(j1, n)

		if !pq.less(j, i) {
			break
		}
		pq.swap(i, j)
		i = j
	}
	return i > i0
}

func (pq priorityQueue[T]) getLesserSibling(lIdx, n int) int {
	ret := lIdx // left child
	if rIdx := lIdx + 1; rIdx < n && pq.less(rIdx, lIdx) {
		ret = rIdx // = 2*i + 2  // right child
	}

	return ret
}

type PriorityQueue[T any] struct {
	priorityQueue[T]
}

func NewPriorityQueue[T any](capacity int) *PriorityQueue[T] {
	return &PriorityQueue[T]{
		priorityQueue: make(priorityQueue[T], 0, capacity),
	}
}

func NewPriorityQueueFromSlice[T any](data []WithPriority[T]) *PriorityQueue[T] {
	pq := priorityQueue[T](data)
	pq.Init()
	return &PriorityQueue[T]{
		priorityQueue: pq,
	}
}

func (pq *PriorityQueue[T]) Push(item T, priority float32) {
	pq.priorityQueue.Push(
		WithPriority[T]{
			Item:     item,
			Priority: priority,
		},
	)
}

func (pq *PriorityQueue[T]) PopWithPriority() (ret WithPriority[T], _ error) {
	if pq.Len() == 0 {
		return ret, errors.New("empty queue")
	}
	item := pq.priorityQueue.Pop()
	return item, nil
}

func (pq *PriorityQueue[T]) Pop() (ret T, _ error) {
	item, err := pq.PopWithPriority()
	if err != nil {
		return ret, err
	}

	return item.Item, nil
}

func (pq *PriorityQueue[T]) PeekWithPriority(n int) (ret WithPriority[T], _ error) {
	if pq.priorityQueue.Len() <= n {
		return ret, fmt.Errorf("index out of range: %d", n)
	}

	item := pq.priorityQueue[n]
	return item, nil
}

func (pq *PriorityQueue[T]) Peek(n int) (ret T, _ error) {
	item, err := pq.PeekWithPriority(n)
	if err != nil {
		return ret, err
	}

	return item.Item, nil
}

func (pq *PriorityQueue[T]) Len() int {
	return pq.priorityQueue.Len()
}
