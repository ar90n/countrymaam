package index

import (
	"context"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"runtime"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/bsp_tree"
	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/sourcegraph/conc/pool"
)

const streamBufferSize = 64
const queueCapcitySize = 64
const defaultTrees = 1

type BspTreeIndex[T linalg.Number] struct {
	Features [][]T
	Trees    []bsp_tree.BspTree[T]
	Dim      uint
}

type queueItem struct {
	RootIdx uint
	NodeIdx uint
}

var _ = (*BspTreeIndex[float32])(nil)

func (bsp BspTreeIndex[T]) SearchChannel(ctx context.Context, query []T) <-chan countrymaam.SearchResult {
	outputStream := make(chan countrymaam.SearchResult, streamBufferSize)
	env := linalg.NewLinAlgFromContext[T](ctx)

	go func() error {
		defer close(outputStream)

		queue := collection.NewPriorityQueue[queueItem](queueCapcitySize)
		for i := range bsp.Trees {
			queue.Push(queueItem{RootIdx: uint(i), NodeIdx: 0}, -math.MaxFloat32)
		}

		for {
			nodeWithPriority, err := queue.PopWithPriority()
			if err != nil {
				if err == collection.ErrEmptyPriorityQueue {
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
					case outputStream <- countrymaam.SearchResult{
						Index:    uint(root.Indice[i]),
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

func (bsp BspTreeIndex[T]) Save(w io.Writer) error {
	return saveIndex(bsp, w)
}

type BspTreeIndexBuilder[T linalg.Number] struct {
	dim            uint
	trees          uint
	maxGoroutines  int
	bspTreeBuilder bsp_tree.BspTreeBuilder[T]
}

func NewBspTreeIndexBuilder[T linalg.Number](dim uint, bspTreeBuilder bsp_tree.BspTreeBuilder[T]) *BspTreeIndexBuilder[T] {
	return &BspTreeIndexBuilder[T]{
		dim:            dim,
		trees:          defaultTrees,
		maxGoroutines:  runtime.NumCPU(),
		bspTreeBuilder: bspTreeBuilder,
	}
}

func (btib *BspTreeIndexBuilder[T]) SetTrees(trees uint) *BspTreeIndexBuilder[T] {
	btib.trees = trees
	return btib
}

func (btib *BspTreeIndexBuilder[T]) SetMaxGoroutines(maxGoroutines uint) *BspTreeIndexBuilder[T] {
	btib.maxGoroutines = int(maxGoroutines)
	return btib
}

func (btib BspTreeIndexBuilder[T]) GetPrameterString() string {
	return fmt.Sprintf("trees=%d_%s", btib.trees, btib.bspTreeBuilder.GetPrameterString())
}

func (btis *BspTreeIndexBuilder[T]) Build(ctx context.Context, features [][]T) (*BspTreeIndex[T], error) {
	bsp_tree.Register[T]()
	gob.Register(BspTreeIndex[T]{})

	env := linalg.NewLinAlgFromContext[T](ctx)

	trees := make([]bsp_tree.BspTree[T], btis.trees)
	p := pool.New().WithMaxGoroutines(btis.maxGoroutines).WithErrors()
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

	index := BspTreeIndex[T]{
		Features: features,
		Trees:    trees,
		Dim:      btis.dim,
	}
	return &index, nil
}

func LoadBspTreeIndex[T linalg.Number](r io.Reader) (*BspTreeIndex[T], error) {
	bsp_tree.Register[T]()
	gob.Register(BspTreeIndex[T]{})

	index, err := loadIndex[BspTreeIndex[T]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}
