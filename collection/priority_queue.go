package collection

import (
	"container/heap"
	"context"
	"errors"
	"fmt"
)

type WithPriority[T any] struct {
	Item     T
	Priority float64
}

type priorityQueue[T any] []*WithPriority[T]

func (pq priorityQueue[T]) Len() int { return len(pq) }

func (pq priorityQueue[T]) Less(i, j int) bool {
	return pq[i].Priority < pq[j].Priority
}

func (pq priorityQueue[T]) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *priorityQueue[T]) Push(x interface{}) {
	item := x.(*WithPriority[T])
	*pq = append(*pq, item)
}

func (pq *priorityQueue[T]) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil // avoid memory leak
	*pq = old[0 : n-1]
	return item
}

type PriorityQueue[T any] struct {
	priorityQueue[T]
}

func NewPriorityQueue[T any](capacity int) *PriorityQueue[T] {
	return &PriorityQueue[T]{
		priorityQueue: make(priorityQueue[T], 0, capacity),
	}
}

func (pq *PriorityQueue[T]) Push(item T, priority float64) {
	heap.Push(
		&pq.priorityQueue,
		&WithPriority[T]{
			Item:     item,
			Priority: priority,
		},
	)
}

func (pq *PriorityQueue[T]) PopWithPriority() (ret WithPriority[T], _ error) {
	if pq.priorityQueue.Len() == 0 {
		return ret, errors.New("empty queue")
	}
	item := heap.Pop(&pq.priorityQueue).(*WithPriority[T])
	return *item, nil
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
	return *item, nil
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
