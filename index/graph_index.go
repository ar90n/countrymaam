package index

import (
	"context"
	"io"
	"math"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/graph"
	"github.com/ar90n/countrymaam/linalg"
)

type GraphIndex[T linalg.Number, U comparable] struct {
	Features [][]T
	Items    []U
	G        graph.Graph
}

func (gi GraphIndex[T, U]) findApproxNearest(entry uint, query []T, env linalg.Env[T]) collection.WithPriority[uint] {
	curIdx := entry
	curDist := env.SqL2(query, gi.Features[curIdx])

	q := collection.NewPriorityQueue[uint](0)
	q.Push(curIdx, curDist)

	best := collection.WithPriority[uint]{
		Item:     uint(0),
		Priority: math.MaxFloat32,
	}
	visited := map[uint]struct{}{curIdx: {}}
	for {
		cur, err := q.PopWithPriority()
		if err != nil {
			break
		}

		if best.Priority < cur.Priority {
			break
		}
		best = cur

		for _, e := range gi.G.Nodes[cur.Item].Neighbors {
			if _, found := visited[e]; found {
				continue
			}
			visited[e] = struct{}{}

			dist := env.SqL2(query, gi.Features[e])
			q.Push(e, dist)
		}
	}

	return best
}

func (gi GraphIndex[T, U]) Search(ctx context.Context, query []T, n uint, maxCandidates uint) ([]countrymaam.Candidate[U], error) {
	ch := gi.SearchChannel(ctx, query)

	items := make([]collection.WithPriority[U], 0, maxCandidates)
	for item := range ch {
		if maxCandidates <= uint(len(items)) {
			break
		}
		items = append(items, collection.WithPriority[U]{Item: item.Item, Priority: item.Distance})
	}
	pq := collection.NewPriorityQueueFromSlice(items)

	// take unique neighbors
	ret := make([]countrymaam.Candidate[U], 0, n)
	founds := make(map[U]struct{}, maxCandidates)
	for uint(len(ret)) < n {
		item, err := pq.PopWithPriority()
		if err != nil {
			break
		}

		if _, ok := founds[item.Item]; ok {
			continue
		}
		founds[item.Item] = struct{}{}

		ret = append(ret, countrymaam.Candidate[U]{Item: item.Item, Distance: item.Priority})
	}
	return ret, nil
}

func (gi GraphIndex[T, U]) SearchChannel(ctx context.Context, query []T) <-chan countrymaam.Candidate[U] {
	env := linalg.NewLinAlgFromContext[T](ctx)
	outputStream := make(chan countrymaam.Candidate[U], streamBufferSize)

	go func() {
		defer close(outputStream)

		approxNearest := gi.findApproxNearest(0, query, linalg.NewLinAlgFromContext[T](ctx))

		q := collection.NewPriorityQueue[uint](0)
		q.Push(approxNearest.Item, approxNearest.Priority)

		visited := map[uint]struct{}{approxNearest.Item: {}}
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
	dim          uint
	graphBuilder graph.GraphBuilder
}

func NewGraphIndexBuilder[T linalg.Number, U comparable](dim uint, graphBuilder graph.GraphBuilder) *GraphIndexBuilder[T, U] {
	creator := GraphIndexBuilder[T, U]{
		dim:          dim,
		graphBuilder: graphBuilder,
	}

	return &creator
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
