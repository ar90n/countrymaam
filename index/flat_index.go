package index

import (
	"math"
	"sort"

	"github.com/ar90n/countrymaam/metric"
	"github.com/ar90n/countrymaam/number"
)

type flatIndex[T number.Number, U any, M metric.Metric[T]] struct {
	dim      uint
	features [][]T
	items    []U
	metric   M
}

var _ = (*flatIndex[float32, int, metric.SqL2Dist[float32]])(nil)

func (fi *flatIndex[T, U, M]) Add(feature []T, item U) {
	fi.features = append(fi.features, feature)
	fi.items = append(fi.items, item)
}

func (fi flatIndex[T, U, M]) Search(query []T, n uint, r float32) ([]U, error) {
	candidates := make([]Candidate[U], n+1)
	for i := range candidates {
		candidates[i].Distance = math.MaxFloat32
	}

	nCandidates := 0
	for i, feature := range fi.features {
		distance := fi.metric.CalcDistance(query, feature)
		if distance < r {
			nCandidates += 1
			candidates[n] = Candidate[U]{
				Distance: distance,
				Item:     fi.items[i],
			}
			sort.Slice(candidates, func(i, j int) bool {
				return candidates[i].Distance < candidates[j].Distance
			})
		}
	}
	if n < uint(nCandidates) {
		nCandidates = int(n)
	}

	results := make([]U, nCandidates)
	for i, c := range candidates[:nCandidates] {
		results[i] = c.Item
	}
	return results, nil
}

func (fi flatIndex[T, U, M]) Build() error {
	return nil
}

func NewFlatIndex[T number.Number, U any, M metric.Metric[T]](dim uint) *flatIndex[T, U, M] {
	return &flatIndex[T, U, M]{
		dim:      dim,
		features: make([][]T, 0),
		items:    make([]U, 0),
	}
}
