package countrymaam

import (
	"context"
	"encoding/gob"
	"errors"
	"io"
	"math"
	"math/rand"
	"sort"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/ar90n/countrymaam/pipeline"
)

type treeElement[T linalg.Number, U comparable] struct {
	Feature []T
	Item    U
}

type treeNode[T linalg.Number, U comparable] struct {
	CutPlane CutPlane[T, U]
	Begin    uint
	End      uint
	Left     *treeNode[T, U]
	Right    *treeNode[T, U]
}

type treeRoot[T linalg.Number, U comparable] struct {
	Indice []int
	Node   *treeNode[T, U]
}

type bspTreeIndex[T linalg.Number, U comparable, C CutPlane[T, U]] struct {
	Dim      uint
	Pool     []treeElement[T, U]
	Roots    []*treeRoot[T, U]
	LeafSize uint
	env      linalg.Env[T]
	nTrees   uint
}

type queueItem[T linalg.Number, U comparable] struct {
	Root *treeRoot[T, U]
	Node *treeNode[T, U]
}

type Result[T any] struct {
	Value T
	Error error
}

var _ = (*bspTreeIndex[float32, int, kdCutPlane[float32, int]])(nil)

func (bsp *bspTreeIndex[T, U, C]) Add(feature []T, item U) {
	bsp.Pool = append(bsp.Pool, treeElement[T, U]{
		Item:    item,
		Feature: feature,
	})
	bsp.ClearIndex()
}

func (bsp *bspTreeIndex[T, U, C]) buildRoot(ctx context.Context) (*treeRoot[T, U], error) {
	indice := pipeline.ToSlice(ctx, pipeline.Seq(ctx, uint(len(bsp.Pool))))
	rand.Shuffle(len(indice), func(i, j int) { indice[i], indice[j] = indice[j], indice[i] })

	node, err := bsp.buildTree(ctx, indice, 0)
	if err != nil {
		return nil, err
	}

	return &treeRoot[T, U]{
		Indice: indice,
		Node:   node,
	}, nil
}

func (bsp *bspTreeIndex[T, U, C]) buildTree(ctx context.Context, indice []int, offset uint) (*treeNode[T, U], error) {
	nElements := uint(len(indice))
	if nElements == 0 {
		return nil, nil
	}

	if nElements <= bsp.LeafSize {
		return &treeNode[T, U]{
			Begin: offset,
			End:   offset + nElements,
		}, nil
	}

	cutPlane, err := (*new(C)).Construct(bsp.Pool, indice, bsp.env)
	if err != nil {
		return nil, err
	}

	mid := collection.Partition(indice, func(i int) bool {
		return cutPlane.Evaluate(bsp.Pool[i].Feature, bsp.env)
	})
	if mid == 0 || mid == uint(len(indice)) {
		mid = uint(len(indice)) / 2
	}

	left, err := bsp.buildTree(ctx, indice[:mid], offset)
	if err != nil {
		return nil, err
	}

	right, err := bsp.buildTree(ctx, indice[mid:], offset+mid)
	if err != nil {
		return nil, err
	}

	return &treeNode[T, U]{
		Begin:    offset,
		End:      offset + nElements,
		Left:     left,
		Right:    right,
		CutPlane: cutPlane,
	}, nil
}

func (bsp *bspTreeIndex[T, U, C]) Build(ctx context.Context) error {
	if bsp.Empty() {
		return errors.New("empty pool")
	}

	taskStream := pipeline.Seq(ctx, bsp.nTrees)
	rootStream := make(chan Result[*treeRoot[T, U]])
	defer close(rootStream)
	//for i := 0; i < runtime.NumCPU(); i++ {
	for i := 0; i < 1; i++ {
		go func() {
			for range taskStream {
				root, err := bsp.buildRoot(ctx)
				rootStream <- Result[*treeRoot[T, U]]{Value: root, Error: err}
			}
		}()
	}

	for uint(len(bsp.Roots)) < bsp.nTrees {
		ret := <-rootStream
		if ret.Error != nil {
			return ret.Error
		}

		bsp.Roots = append(bsp.Roots, ret.Value)
	}
	return nil
}

type Candidates[U comparable] []Candidate[U]

func (c Candidates[U]) Len() int {
	return len(c)
}
func (c Candidates[U]) Swap(i, j int) {
	c[i], c[j] = c[j], c[i]
}
func (c Candidates[U]) Less(i, j int) bool {
	return c[i].Distance < c[j].Distance
}

func (bsp *bspTreeIndex[T, U, C]) Search(ctx context.Context, query []T, n uint, maxCandidates uint) ([]Candidate[U], error) {
	ch := bsp.SearchChannel(ctx, query)

	items := make([]Candidate[U], 0, maxCandidates)
	founds := make(map[U]struct{}, maxCandidates)
	i := uint(0)
	for item := range ch {
		if maxCandidates < i {
			break
		}
		i++

		if _, ok := founds[item.Item]; ok {
			continue
		}
		founds[item.Item] = struct{}{}

		items = append(items, item)
	}

	sort.Sort(Candidates[U](items))
	if uint(len(items)) < n {
		n = uint(len(items))
	}
	return items[:n], nil
}

func (bsp *bspTreeIndex[T, U, C]) SearchChannel(ctx context.Context, query []T) <-chan Candidate[U] {
	outputStream := make(chan Candidate[U], 64)
	go func() error {
		defer close(outputStream)

		if !bsp.HasIndex() {
			bsp.Build(ctx)
		}

		maxCandidates := 64
		queue := collection.NewPriorityQueue[queueItem[T, U]](maxCandidates)
		for i := range bsp.Roots {
			queue.Push(queueItem[T, U]{Root: bsp.Roots[i], Node: bsp.Roots[i].Node}, -math.MaxFloat64)
		}

		for 0 < queue.Len() {
			nodeWithPriority, err := queue.PopWithPriority()
			if err != nil {
				continue
			}

			node := nodeWithPriority.Item.Node
			if node == nil {
				continue
			}

			root := nodeWithPriority.Item.Root
			if node.Left == nil && node.Right == nil {
				for i := node.Begin; i < node.End; i++ {
					elem := bsp.Pool[root.Indice[i]]
					distance := float64(bsp.env.SqL2(query, elem.Feature))
					select {
					case <-ctx.Done():
						return nil
					case outputStream <- Candidate[U]{
						Item:     elem.Item,
						Distance: distance,
					}:
					}
				}
			} else {
				worstPriority := nodeWithPriority.Priority
				sqDistanceToCutPlane := node.CutPlane.Distance(query, bsp.env)
				rightPriority := linalg.Max(-sqDistanceToCutPlane, worstPriority)
				queue.Push(queueItem[T, U]{Root: root, Node: node.Right}, rightPriority)

				leftPriority := linalg.Max(sqDistanceToCutPlane, worstPriority)
				queue.Push(queueItem[T, U]{Root: root, Node: node.Left}, leftPriority)
			}
		}

		return nil
	}()

	return outputStream
}

func (bsp bspTreeIndex[T, U, C]) HasIndex() bool {
	return 0 < len(bsp.Roots)
}

func (bsp *bspTreeIndex[T, U, C]) ClearIndex() {
	bsp.Roots = nil
}

func (bsp bspTreeIndex[T, U, C]) Empty() bool {
	return len(bsp.Pool) == 0
}

func (bsp bspTreeIndex[T, U, C]) Save(w io.Writer) error {
	return saveIndex(&bsp, w)
}

func NewKdTreeIndex[T linalg.Number, U comparable](dim uint, leafSize uint, opts linalg.LinAlgOptions) *bspTreeIndex[T, U, kdCutPlane[T, U]] {
	gob.Register(kdCutPlane[T, U]{})

	env := linalg.NewLinAlg[T](opts)
	return &bspTreeIndex[T, U, kdCutPlane[T, U]]{
		Dim:      dim,
		Pool:     make([]treeElement[T, U], 0, 4096),
		LeafSize: leafSize,
		env:      env,
		nTrees:   1,
	}
}

func LoadKdTreeIndex[T linalg.Number, U comparable](r io.Reader, opts linalg.LinAlgOptions) (*bspTreeIndex[T, U, kdCutPlane[T, U]], error) {
	gob.Register(kdCutPlane[T, U]{})
	index, err := loadIndex[bspTreeIndex[T, U, kdCutPlane[T, U]]](r)
	if err != nil {
		return nil, err
	}
	index.env = linalg.NewLinAlg[T](opts)

	return &index, nil
}

func NewRpTreeIndex[T linalg.Number, U comparable](dim uint, leafSize uint, opts linalg.LinAlgOptions) *bspTreeIndex[T, U, rpCutPlane[T, U]] {
	gob.Register(rpCutPlane[T, U]{})

	env := linalg.NewLinAlg[T](opts)
	return &bspTreeIndex[T, U, rpCutPlane[T, U]]{
		Dim:      dim,
		Pool:     make([]treeElement[T, U], 0),
		LeafSize: leafSize,
		env:      env,
		nTrees:   1,
	}
}

func LoadRpTreeIndex[T linalg.Number, U comparable](r io.Reader, opts linalg.LinAlgOptions) (*bspTreeIndex[T, U, rpCutPlane[T, U]], error) {
	gob.Register(rpCutPlane[T, U]{})
	index, err := loadIndex[bspTreeIndex[T, U, rpCutPlane[T, U]]](r)
	if err != nil {
		return nil, err
	}
	index.env = linalg.NewLinAlg[T](opts)

	return &index, nil
}

func NewRandomizedKdTreeIndex[T linalg.Number, U comparable](dim uint, leafSize uint, nTrees uint, opts linalg.LinAlgOptions) *bspTreeIndex[T, U, randomizedKdCutPlane[T, U]] {
	gob.Register(kdCutPlane[T, U]{})
	gob.Register(randomizedKdCutPlane[T, U]{})

	env := linalg.NewLinAlg[T](opts)
	return &bspTreeIndex[T, U, randomizedKdCutPlane[T, U]]{
		Dim:      dim,
		Pool:     make([]treeElement[T, U], 0),
		LeafSize: leafSize,
		env:      env,
		nTrees:   nTrees,
	}
}

func LoadRandomizedKdTreeIndex[T linalg.Number, U comparable](r io.Reader, opts linalg.LinAlgOptions) (*bspTreeIndex[T, U, randomizedKdCutPlane[T, U]], error) {
	gob.Register(kdCutPlane[T, U]{})
	gob.Register(randomizedKdCutPlane[T, U]{})
	index, err := loadIndex[bspTreeIndex[T, U, randomizedKdCutPlane[T, U]]](r)
	if err != nil {
		return nil, err
	}
	index.env = linalg.NewLinAlg[T](opts)

	return &index, nil
}

func NewRandomizedRpTreeIndex[T linalg.Number, U comparable](dim uint, leafSize uint, nTrees uint, opts linalg.LinAlgOptions) *bspTreeIndex[T, U, rpCutPlane[T, U]] {
	gob.Register(rpCutPlane[T, U]{})

	env := linalg.NewLinAlg[T](opts)
	return &bspTreeIndex[T, U, rpCutPlane[T, U]]{
		Dim:      dim,
		Pool:     make([]treeElement[T, U], 0, 4096),
		LeafSize: leafSize,
		env:      env,
		nTrees:   nTrees,
	}
}

func LoadRandomizedRpTreeIndex[T linalg.Number, U comparable](r io.Reader, opts linalg.LinAlgOptions) (*bspTreeIndex[T, U, rpCutPlane[T, U]], error) {
	gob.Register(rpCutPlane[T, U]{})
	index, err := loadIndex[bspTreeIndex[T, U, rpCutPlane[T, U]]](r)
	if err != nil {
		return nil, err
	}
	index.env = linalg.NewLinAlg[T](opts)

	return &index, nil
}
