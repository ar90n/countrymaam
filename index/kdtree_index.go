package index

import (
	"errors"
	"math"
	"sort"

	my_constraints "github.com/ar90n/countrymaam/constraints"
	"github.com/ar90n/countrymaam/metric"
)

type kdCutPlane[T my_constraints.Number] struct {
	Axis  uint
	Value T
}

type kdElement[T my_constraints.Number, U any] struct {
	Feature []T
	Item    U
}

type kdNode[T my_constraints.Number, U any] struct {
	CutPlane kdCutPlane[T]
	Begin    uint
	End      uint
	Left     *kdNode[T, U]
	Right    *kdNode[T, U]
}

type kdTreeIndex[T my_constraints.Number, U any, M metric.Metric[T]] struct {
	Dim    uint
	Pool   []*kdElement[T, U]
	Metric M
	Root   *kdNode[T, U]
}

func (ki *kdTreeIndex[T, U, M]) Add(feature []T, item U) {
	ki.Pool = append(ki.Pool, &kdElement[T, U]{
		Feature: feature,
		Item:    item,
	})
}

func (fi kdTreeIndex[T, U, M]) search(node *kdNode[T, U], query []T, n uint, r float32) []Candidate[U] {
	candidates := make([]Candidate[U], 0)
	if node.Left == nil && node.Right == nil {
		for i := node.Begin; i < node.End; i++ {
			distance := fi.Metric.CalcDistance(query, fi.Pool[i].Feature)
			if distance < r {
				candidates = append(candidates, Candidate[U]{distance, fi.Pool[i].Item})
			}
		}

		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Distance < candidates[j].Distance
		})
		if n < uint(len(candidates)) {
			candidates = candidates[:n]
		}
		return candidates
	}

	if query[node.CutPlane.Axis] <= node.CutPlane.Value {
		if node.Left != nil {
			leftCandidates := fi.search(node.Left, query, n, r)
			candidates = append(candidates, leftCandidates...)
		}
	} else {
		if node.Right != nil {
			rightCandidates := fi.search(node.Right, query, n, r)
			candidates = append(candidates, rightCandidates...)
		}
	}
	if len(candidates) == 0 {
		return candidates
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].Distance < candidates[j].Distance
	})
	if n < uint(len(candidates)) {
		candidates = candidates[:n]
	}

	maxDistance := candidates[len(candidates)-1].Distance
	distanceToCutPlane := float32(math.Abs(float64(query[node.CutPlane.Axis] - node.CutPlane.Value)))
	if distanceToCutPlane < maxDistance || len(candidates) < int(n) {
		if query[node.CutPlane.Axis] <= node.CutPlane.Value {
			if node.Right != nil {
				rightCandidates := fi.search(node.Right, query, n, r)
				candidates = append(candidates, rightCandidates...)
			}
		} else {
			if node.Left != nil {
				leftCandidates := fi.search(node.Left, query, n, r)
				candidates = append(candidates, leftCandidates...)
			}
		}
		sort.Slice(candidates, func(i, j int) bool {
			return candidates[i].Distance < candidates[j].Distance
		})
		if n < uint(len(candidates)) {
			candidates = candidates[:n]
		}
	}

	return candidates
}

func (ki kdTreeIndex[T, U, M]) Search(query []T, n uint, r float32) []U {
	candidates := ki.search(ki.Root, query, n, r)
	results := make([]U, len(candidates))
	for i, c := range candidates {
		results[i] = c.Item
	}
	return results
}

func makeCutPlane[T my_constraints.Number, U any](pool []*kdElement[T, U], begin uint, end uint) (kdCutPlane[T], error) {
	if len(pool) == 0 {
		return kdCutPlane[T]{}, errors.New("empty pool")
	}

	if begin == end {
		return kdCutPlane[T]{}, errors.New("begin == end")
	}

	minValues := append([]T{}, pool[begin].Feature...)
	maxValues := append([]T{}, pool[begin].Feature...)
	for i := begin + 1; i < end; i++ {
		for j := range pool[i].Feature {
			if pool[i].Feature[j] < minValues[j] {
				minValues[j] = pool[i].Feature[j]
			}
			if pool[i].Feature[j] > maxValues[j] {
				maxValues[j] = pool[i].Feature[j]
			}
		}
	}

	maxRange := maxValues[0] - minValues[0]
	axis := uint(0)
	value := (maxValues[0] + minValues[0]) / 2
	for i := axis + 1; i < uint(len(minValues)); i++ {
		diff := maxValues[i] - minValues[i]
		if maxRange < diff {
			maxRange = diff
			axis = i
			value = (maxValues[i] + minValues[i]) / 2
		}
	}

	cutPlane := kdCutPlane[T]{
		Axis:  axis,
		Value: value,
	}
	return cutPlane, nil
}

func (ki *kdTreeIndex[T, U, M]) Build() error {
	if len(ki.Pool) == 0 {
		return errors.New("empty pool")
	}

	root, err := ki.build(0, uint(len(ki.Pool)))
	if err != nil {
		return errors.New("build failed")
	}
	ki.Root = root
	return nil
}

func (ki *kdTreeIndex[T, U, M]) build(begin uint, end uint) (*kdNode[T, U], error) {
	if begin == end {
		return nil, nil
	}

	if (end - begin) <= 1 {
		return &kdNode[T, U]{
			Begin: begin,
			End:   end,
		}, nil
	}

	cutPlane, err := makeCutPlane(ki.Pool, begin, end)
	if err != nil {
		return nil, err
	}

	midIndex := partition(ki.Pool, begin, end, cutPlane)
	left, err := ki.build(begin, midIndex)
	if err != nil {
		return nil, err
	}
	right, err := ki.build(midIndex, end)
	if err != nil {
		return nil, err
	}

	return &kdNode[T, U]{
		Begin:    begin,
		End:      end,
		Left:     left,
		Right:    right,
		CutPlane: cutPlane,
	}, nil
}

func partition[T my_constraints.Number, U any](pool []*kdElement[T, U], begin uint, end uint, cutPlane kdCutPlane[T]) uint {
	i, j := begin, end-1
	for i <= j {
		for i <= j && pool[i].Feature[cutPlane.Axis] <= cutPlane.Value {
			i++
		}
		for i <= j && pool[j].Feature[cutPlane.Axis] > cutPlane.Value {
			j--
		}
		if i < j {
			pool[i], pool[j] = pool[j], pool[i]
		}
	}
	return i
}

func NewKDTreeIndex[T my_constraints.Number, U any, M metric.Metric[T]](dim uint) *kdTreeIndex[T, U, M] {
	return &kdTreeIndex[T, U, M]{
		Dim: dim,
	}
}
