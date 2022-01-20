package countrymaam

import (
	"io"
	"math"
	"sort"

	"github.com/ar90n/countrymaam/number"
)

type flatIndex[T number.Number, U any] struct {
	Dim      uint
	Features [][]T
	Items    []U
}

var _ = (*flatIndex[float32, int])(nil)

func (fi *flatIndex[T, U]) Add(feature []T, item U) {
	fi.Features = append(fi.Features, feature)
	fi.Items = append(fi.Items, item)
}

func (fi flatIndex[T, U]) Search(query []T, n uint, r float64) ([]Candidate[U], error) {
	candidates := make([]Candidate[U], n+1)
	for i := range candidates {
		candidates[i].Distance = math.MaxFloat32
	}

	nCandidates := uint(0)
	for i, feature := range fi.Features {
		distance := number.CalcSqDistance(query, feature)
		if distance < r {
			nCandidates += 1
			candidates[n] = Candidate[U]{
				Distance: distance,
				Item:     fi.Items[i],
			}
			sort.Slice(candidates, func(i, j int) bool {
				return candidates[i].Distance < candidates[j].Distance
			})
		}
	}
	if n < nCandidates {
		nCandidates = n
	}

	return candidates[:nCandidates], nil
}

func (fi flatIndex[T, U]) Build() error {
	return nil
}

func (fi flatIndex[T, U]) Save(w io.Writer) error {
	return saveIndex(&fi, w)
}
