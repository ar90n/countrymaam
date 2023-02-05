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
	Left     uint
	Right    uint
}

type treeRoot[T linalg.Number, U comparable] struct {
	Indice []int
	Nodes  []treeNode[T, U]
}

type bspTreeIndex[T linalg.Number, U comparable, C CutPlane[T, U]] struct {
	Dim      uint
	Pool     []treeElement[T, U]
	Roots    []treeRoot[T, U]
	LeafSize uint
	env      linalg.Env[T]
	nTrees   uint
}

type queueItem[T linalg.Number, U comparable] struct {
	Root uint
	Node uint
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

func (bsp *bspTreeIndex[T, U, C]) buildRoot(ctx context.Context, id uint) (treeRoot[T, U], error) {
	indice := pipeline.ToSlice(ctx, pipeline.Seq(ctx, uint(len(bsp.Pool))))
	rand.Shuffle(len(indice), func(i, j int) { indice[i], indice[j] = indice[j], indice[i] })

	bsp.Roots[id] = treeRoot[T, U]{
		Indice: indice,
		Nodes: []treeNode[T, U]{
			{
				Begin: 0,
				End:   uint(len(indice)),
			},
		},
	}
	node, err := bsp.buildTree(ctx, indice, 0, id)
	if err != nil {
		return treeRoot[T, U]{}, err
	}
	bsp.Roots[id].Nodes[0] = bsp.Roots[id].Nodes[node]

	return bsp.Roots[id], nil
}

func (bsp *bspTreeIndex[T, U, C]) buildTree(ctx context.Context, indice []int, offset uint, id uint) (uint, error) {
	nElements := uint(len(indice))
	if nElements == 0 {
		return 0, nil
	}

	if nElements <= bsp.LeafSize {
		node := treeNode[T, U]{
			Begin: offset,
			End:   offset + nElements,
		}
		nNodes := uint(len(bsp.Roots[id].Nodes))
		bsp.Roots[id].Nodes = append(bsp.Roots[id].Nodes, node)
		return nNodes, nil
	}

	cutPlane, err := (*new(C)).Construct(bsp.Pool, indice, bsp.env)
	if err != nil {
		return 0, err
	}

	mid := collection.Partition(indice, func(i int) bool {
		return cutPlane.Evaluate(bsp.Pool[i].Feature, bsp.env)
	})
	if mid == 0 || mid == uint(len(indice)) {
		mid = uint(len(indice)) / 2
	}

	left, err := bsp.buildTree(ctx, indice[:mid], offset, id)
	if err != nil {
		return 0, err
	}

	right, err := bsp.buildTree(ctx, indice[mid:], offset+mid, id)
	if err != nil {
		return 0, err
	}

	node := treeNode[T, U]{
		Begin:    offset,
		End:      offset + nElements,
		Left:     left,
		Right:    right,
		CutPlane: cutPlane,
	}
	nNodes := uint(len(bsp.Roots[id].Nodes))
	bsp.Roots[id].Nodes = append(bsp.Roots[id].Nodes, node)
	return nNodes, nil
}

func (bsp *bspTreeIndex[T, U, C]) Build(ctx context.Context) error {
	if bsp.Empty() {
		return errors.New("empty pool")
	}

	taskStream := pipeline.Seq(ctx, bsp.nTrees)
	rootStream := make(chan Result[treeRoot[T, U]])
	defer close(rootStream)

	bsp.Roots = make([]treeRoot[T, U], bsp.nTrees)
	//for i := 0; i < runtime.NumCPU(); i++ {
	for i := 0; i < 1; i++ {
		go func() {
			for t := range taskStream {
				root, err := bsp.buildRoot(ctx, uint(t))
				rootStream <- Result[treeRoot[T, U]]{Value: root, Error: err}
			}
		}()
	}

	i := 0
	for uint(i) < bsp.nTrees {
		ret := <-rootStream
		if ret.Error != nil {
			return ret.Error
		}

		bsp.Roots[i] = ret.Value
		i++
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
			queue.Push(queueItem[T, U]{Root: uint(i), Node: 0}, -math.MaxFloat64)
		}

		for 0 < queue.Len() {
			nodeWithPriority, err := queue.PopWithPriority()
			if err != nil {
				continue
			}

			ri := nodeWithPriority.Item.Root
			node := bsp.Roots[ri].Nodes[nodeWithPriority.Item.Node]
			if node.Left == 0 && node.Right == 0 {
				for i := node.Begin; i < node.End; i++ {
					elem := bsp.Pool[bsp.Roots[ri].Indice[i]]
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
				continue
			}

			worstPriority := nodeWithPriority.Priority
			sqDistanceToCutPlane := node.CutPlane.Distance(query, bsp.env)
			if 0 < node.Right {
				rightPriority := linalg.Max(-sqDistanceToCutPlane, worstPriority)
				queue.Push(queueItem[T, U]{Root: ri, Node: node.Right}, rightPriority)
			}

			if 0 < node.Left {
				leftPriority := linalg.Max(sqDistanceToCutPlane, worstPriority)
				queue.Push(queueItem[T, U]{Root: ri, Node: node.Left}, leftPriority)
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
