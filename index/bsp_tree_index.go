package index

import (
	"context"
	"io"
	"math"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/bsp_tree"
	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/sourcegraph/conc/pool"
)

const streamBufferSize = 64
const queueCapcitySize = 64
const defaultTrees = 1

type BspTreeIndex[T linalg.Number, U comparable] struct {
	Features [][]T
	Items    []U
	Trees    []bsp_tree.BspTree[T]
	Dim      uint
}

type queueItem struct {
	RootIdx uint
	NodeIdx uint
}

var _ = (*BspTreeIndex[float32, int])(nil)

func (bsp BspTreeIndex[T, U]) Search(ctx context.Context, query []T, n uint, maxCandidates uint) ([]countrymaam.Candidate[U], error) {
	ch := bsp.SearchChannel(ctx, query)

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
			return nil, err
		}

		if _, ok := founds[item.Item]; ok {
			continue
		}
		founds[item.Item] = struct{}{}

		ret = append(ret, countrymaam.Candidate[U]{Item: item.Item, Distance: item.Priority})
	}
	return ret, nil
}

func (bsp BspTreeIndex[T, U]) SearchChannel(ctx context.Context, query []T) <-chan countrymaam.Candidate[U] {
	outputStream := make(chan countrymaam.Candidate[U], streamBufferSize)
	go func() error {
		defer close(outputStream)

		env := linalg.NewLinAlgFromContext[T](ctx)

		queue := collection.NewPriorityQueue[queueItem](queueCapcitySize)
		for i := range bsp.Trees {
			queue.Push(queueItem{RootIdx: uint(i), NodeIdx: 0}, -math.MaxFloat32)
		}

		for {
			nodeWithPriority, err := queue.PopWithPriority()
			if err != nil {
				if queue.Len() == 0 {
					break
				}

				continue
			}

			ri := nodeWithPriority.Item.RootIdx
			root := bsp.Trees[ri]
			node := root.Nodes[nodeWithPriority.Item.NodeIdx]
			if node.Left == 0 && node.Right == 0 {
				for i := node.Begin; i < node.End; i++ {
					feature := bsp.Features[root.Indice[i]]
					distance := env.SqL2(query, feature)
					select {
					case <-ctx.Done():
						return nil
					case outputStream <- countrymaam.Candidate[U]{
						Item:     bsp.Items[root.Indice[i]],
						Distance: distance,
					}:
					}
				}
				continue
			}

			worstPriority := nodeWithPriority.Priority
			sqDistanceToCutPlane := float32(node.CutPlane.Distance(query, env))
			if 0 < node.Right {
				rightPriority := linalg.Max(-sqDistanceToCutPlane, worstPriority)
				queue.Push(queueItem{RootIdx: ri, NodeIdx: node.Right}, rightPriority)
			}

			if 0 < node.Left {
				leftPriority := linalg.Max(sqDistanceToCutPlane, worstPriority)
				queue.Push(queueItem{RootIdx: ri, NodeIdx: node.Left}, leftPriority)
			}
		}

		return nil
	}()

	return outputStream
}

func (bsp BspTreeIndex[T, U]) Save(w io.Writer) error {
	return saveIndex(bsp, w)
}

type BspTreeIndexBuilder[T linalg.Number, U comparable] struct {
	dim            uint
	trees          uint
	bspTreeBuilder bsp_tree.BspTreeBuilder[T]
}

func NewBspTreeIndexBuilder[T linalg.Number, U comparable](dim uint, bspTreeBuilder bsp_tree.BspTreeBuilder[T]) *BspTreeIndexBuilder[T, U] {
	return &BspTreeIndexBuilder[T, U]{
		dim:            dim,
		trees:          defaultTrees,
		bspTreeBuilder: bspTreeBuilder,
	}
}

func (btib *BspTreeIndexBuilder[T, U]) Trees(trees uint) *BspTreeIndexBuilder[T, U] {
	btib.trees = trees
	return btib
}

func (btis *BspTreeIndexBuilder[T, U]) Build(ctx context.Context, features [][]T, items []U) (countrymaam.Index[T, U], error) {
	bsp_tree.Register[T]()

	env := linalg.NewLinAlgFromContext[T](ctx)

	procs := 1
	trees := make([]bsp_tree.BspTree[T], btis.trees)
	p := pool.New().WithMaxGoroutines(int(procs)).WithErrors()
	for i := uint(0); i < btis.trees; i++ {
		i := i
		p.Go(func() error {
			root, err := btis.bspTreeBuilder.Build(features, env)
			if err != nil {
				return err
			}
			trees[i] = root
			return nil
		})
	}

	err := p.Wait()
	if err != nil {
		return nil, err
	}

	index := BspTreeIndex[T, U]{
		Features: features,
		Items:    items,
		Trees:    trees,
		Dim:      btis.dim,
	}
	return &index, nil
}

func LoadBspTreeIndex[T linalg.Number, U comparable](r io.Reader) (*BspTreeIndex[T, U], error) {
	bsp_tree.Register[T]()

	index, err := loadIndex[BspTreeIndex[T, U]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}
