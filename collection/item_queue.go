package collection

import (
	"errors"
	"math"
)

type ItemQueue[T comparable] struct {
	items  []WithPriority[T]
	set    map[T]interface{}
	origin int
}

func NewItemQueue[T comparable](maxSize int) *ItemQueue[T] {
	items := make([]WithPriority[T], maxSize+1)
	for i := range items {
		items[i].Priority = math.MaxFloat32
	}
	return &ItemQueue[T]{
		items:  items,
		set:    make(map[T]interface{}),
		origin: 0,
	}
}

func prevIndex(i, n int) int {
	return (i + n - 1) % n
}

func nextIndex(i, n int) int {
	return (i + 1) % n
}

func (iq *ItemQueue[T]) Push(item T, priority float32) {
	ind := prevIndex(iq.origin, len(iq.items))
	iq.items[ind] = WithPriority[T]{
		Item:     item,
		Priority: priority,
	}
	defer func(ind int) {
		iq.items[ind].Priority = math.MaxFloat32
	}(ind)

	if _, ok := iq.set[item]; ok {
		return
	}
	iq.set[item] = struct{}{}

	nextInd := prevIndex(ind, len(iq.items))
	for (iq.items[ind].Priority < iq.items[nextInd].Priority) && (ind != iq.origin) {
		iq.items[ind], iq.items[nextInd] = iq.items[nextInd], iq.items[ind]
		ind = nextInd
		nextInd = prevIndex(ind, len(iq.items))
	}
}

func (iq *ItemQueue[T]) Pop() (item WithPriority[T], _ error) {
	if iq.items[iq.origin].Priority == math.MaxFloat32 {
		return item, errors.New("queue is empty")
	}

	item = iq.items[iq.origin]
	iq.items[iq.origin].Priority = math.MaxFloat32
	delete(iq.set, item.Item)
	iq.origin = nextIndex(iq.origin, len(iq.items))
	return item, nil
}

func (iq ItemQueue[T]) WorstPriority() float32 {
	ind := prevIndex(iq.origin, len(iq.items))
	ind = prevIndex(ind, len(iq.items))
	return iq.items[ind].Priority
}

func (iq ItemQueue[T]) Len() int {
	beg := iq.origin
	end := prevIndex(beg, len(iq.items))

	n := 0
	for beg != end {
		if iq.items[beg].Priority == math.MaxFloat32 {
			break
		}
		n++
		beg = nextIndex(beg, len(iq.items))
	}
	return n
}
