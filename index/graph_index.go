package index

import (
	"context"
	"errors"
	"io"
	"math/rand"
	"runtime"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/graph"
	"github.com/ar90n/countrymaam/linalg"
)

const defaultEntriesNum = 10

type GraphIndex[T linalg.Number, U comparable] struct {
	Features [][]T
	Items    []U
	G        graph.Graph
}

func (gi GraphIndex[T, U]) findApproxNearest(entry uint, distFunc func(i uint) float32) (collection.WithPriority[uint], error) {
	if uint(len(gi.Features)) <= entry {
		return collection.WithPriority[uint]{}, errors.New("entry is out of range")
	}

	bestIdx := entry
	bestDist := distFunc(bestIdx)

	visited := map[uint]struct{}{bestIdx: {}}
	for {
		isChanged := false
		candidates := gi.G.Nodes[bestIdx].Neighbors
		for _, candIdx := range candidates {
			if _, found := visited[candIdx]; found {
				continue
			}
			visited[candIdx] = struct{}{}

			candDist := distFunc(candIdx)
			if candDist < bestDist {
				bestIdx = candIdx
				bestDist = candDist
				isChanged = true
			}
		}

		if !isChanged {
			break
		}
	}

	//fmt.Fprintln(os.Stderr, "bestIdx:", bestIdx, "bestDist:", bestDist)
	return collection.WithPriority[uint]{
		Item:     bestIdx,
		Priority: bestDist,
	}, nil
}

func (gi GraphIndex[T, U]) SearchChannel(ctx context.Context, query []T) <-chan countrymaam.Candidate[U] {
	entries := make([]uint, defaultEntriesNum)
	for i := range entries {
		entries[i] = uint(rand.Intn(len(gi.Features)))
	}

	return gi.SearchChannelWithEntries(ctx, query, entries)
}

func (gi GraphIndex[T, U]) SearchChannelWithEntries(ctx context.Context, query []T, entries []uint) <-chan countrymaam.Candidate[U] {
	env := linalg.NewLinAlgFromContext[T](ctx)
	distFunc := func(i uint) float32 {
		return env.SqL2(query, gi.Features[i])
	}
	outputStream := make(chan countrymaam.Candidate[U], streamBufferSize)

	go func() {
		defer close(outputStream)

		q := collection.NewPriorityQueue[uint](0)
		visited := map[uint]struct{}{}
		for _, entry := range entries {
			approxNearest, err := gi.findApproxNearest(entry, distFunc)
			if err != nil {
				continue
			}

			if _, found := visited[approxNearest.Item]; found {
				continue
			}
			visited[approxNearest.Item] = struct{}{}
			q.Push(approxNearest.Item, approxNearest.Priority)
		}

		//visited := make([]bool, len(gi.Features))
		for {
			cur, err := q.PopWithPriority()
			if err != nil {
				return
			}

			select {
			case <-ctx.Done():
				return
			case outputStream <- countrymaam.Candidate[U]{
				Item:     gi.Items[cur.Item],
				Distance: cur.Priority,
			}:
			}

			for _, e := range gi.G.Nodes[cur.Item].Neighbors {
				if _, found := visited[e]; found {
					continue
				}
				visited[e] = struct{}{}
				//if visited[e] {
				//	continue
				//}
				//visited[e] = true

				dist := env.SqL2(query, gi.Features[e])
				q.Push(e, dist)
			}
		}
	}()

	return outputStream
}

func (gi GraphIndex[T, U]) Save(w io.Writer) error {
	return saveIndex(gi, w)
}

type GraphIndexBuilder[T linalg.Number, U comparable] struct {
	dim           uint
	maxGoroutines int
	graphBuilder  graph.GraphBuilder
}

func NewGraphIndexBuilder[T linalg.Number, U comparable](dim uint, graphBuilder graph.GraphBuilder) *GraphIndexBuilder[T, U] {
	creator := GraphIndexBuilder[T, U]{
		dim:           dim,
		maxGoroutines: runtime.NumCPU(),
		graphBuilder:  graphBuilder,
	}

	return &creator
}

func (agib *GraphIndexBuilder[T, U]) SetMaxGoroutines(maxGoroutines uint) {
	agib.maxGoroutines = int(maxGoroutines)
}

func (agib *GraphIndexBuilder[T, U]) Build(ctx context.Context, features [][]T, items []U) (countrymaam.Index[T, U], error) {
	graph.Register[T]()

	env := linalg.NewLinAlgFromContext[T](ctx)
	g, err := agib.graphBuilder.Build(
		uint(len(features)),
		func(i, j uint) float32 {
			return env.SqL2(features[j], features[j])
		})
	if err != nil {
		return nil, err
	}

	g = graph.ConvertToUndirected(g)

	return &GraphIndex[T, U]{
		Features: features,
		Items:    items,
		G:        g,
	}, nil
}

func LoadGraphIndex[T linalg.Number, U comparable](r io.Reader) (*GraphIndex[T, U], error) {
	graph.Register[T]()

	index, err := loadIndex[GraphIndex[T, U]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}
