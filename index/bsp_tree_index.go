package index

import (
	"context"
	"encoding/gob"
	"errors"
	"io"
	"log"
	"math"
	"math/rand"
	"runtime"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/sourcegraph/conc/pool"
)

const streamBufferSize = 64
const queueCapcitySize = 64
const defaultLeafs = 16

type CutPlaneFactory[T linalg.Number, U comparable] interface {
	Default() CutPlane[T, U]
	Build(elements []TreeElement[T, U], indice []int, env linalg.Env[T]) (CutPlane[T, U], error)
}

type CutPlane[T linalg.Number, U comparable] interface {
	Evaluate(feature []T, env linalg.Env[T]) bool
	Distance(feature []T, env linalg.Env[T]) float64
}

type TreeElement[T linalg.Number, U comparable] struct {
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

func (r *treeRoot[T, U]) addNode(node treeNode[T, U]) uint {
	nc := uint(len(r.Nodes))
	r.Nodes = append(r.Nodes, node)

	return nc
}

func (r *treeRoot[T, U]) Build(elements []TreeElement[T, U], indice []int, leafs, offset uint, env linalg.Env[T], cf CutPlaneFactory[T, U]) (uint, error) {
	ec := uint(len(indice))
	if ec == 0 {
		return 0, nil
	}

	curIdx := r.addNode(treeNode[T, U]{
		Begin: offset,
		End:   offset + ec,
	})

	if ec <= leafs {
		return curIdx, nil
	}

	cutPlane, err := cf.Build(elements, indice, env)
	if err != nil {
		return 0, err
	}
	r.Nodes[curIdx].CutPlane = cutPlane

	mid := collection.Partition(indice, func(i int) bool {
		return cutPlane.Evaluate(elements[i].Feature, env)
	})

	left, err := r.Build(elements, indice[:mid], leafs, offset, env, cf)
	if err != nil {
		return 0, err
	}
	r.Nodes[curIdx].Left = left

	right, err := r.Build(elements, indice[mid:], leafs, offset+mid, env, cf)
	if err != nil {
		return 0, err
	}
	r.Nodes[curIdx].Right = right

	return curIdx, nil
}

func newTreeRoot[T linalg.Number, U comparable](elements []TreeElement[T, U], leafs uint, env linalg.Env[T], cpf CutPlaneFactory[T, U]) (treeRoot[T, U], error) {
	indice := make([]int, len(elements))
	for i := range indice {
		indice[i] = i
	}
	rand.Shuffle(len(indice), func(i, j int) { indice[i], indice[j] = indice[j], indice[i] })

	root := treeRoot[T, U]{
		Indice: indice,
		Nodes:  []treeNode[T, U]{},
	}
	_, err := root.Build(elements, indice, leafs, 0, env, cpf)
	if err != nil {
		return root, err
	}

	return root, nil
}

type TreeConfig struct {
	Dim   uint
	Leafs uint
	Trees uint
	Procs uint
}

type bspTreeIndex[T linalg.Number, U comparable] struct {
	Elements []TreeElement[T, U]
	Roots    []treeRoot[T, U]
	cpf      CutPlaneFactory[T, U]
	Dim      uint
	Leafs    uint
	Trees    uint
	Procs    uint
}

type queueItem struct {
	RootIdx uint
	NodeIdx uint
}

type result[T any] struct {
	Value T
	Error error
}

var _ = (*bspTreeIndex[float32, int])(nil)

func (bsp *bspTreeIndex[T, U]) Add(feature []T, item U) {
	bsp.Elements = append(bsp.Elements, TreeElement[T, U]{
		Item:    item,
		Feature: feature,
	})
	bsp.ClearIndex()
}

func (bsp *bspTreeIndex[T, U]) Build(ctx context.Context) error {
	if bsp.Empty() {
		return errors.New("empty pool")
	}

	env := linalg.NewLinAlgFromContext[T](ctx)

	bsp.Roots = make([]treeRoot[T, U], bsp.Trees)
	p := pool.New().WithMaxGoroutines(int(bsp.Procs)).WithErrors()
	for i := uint(0); i < bsp.Trees; i++ {
		i := i
		p.Go(func() error {
			root, err := newTreeRoot(bsp.Elements, bsp.Leafs, env, bsp.cpf)
			if err != nil {
				return err
			}
			bsp.Roots[i] = root
			return nil
		})
	}

	return p.Wait()
}

func (bsp *bspTreeIndex[T, U]) Search(ctx context.Context, query []T, n uint, maxCandidates uint) ([]Candidate[U], error) {
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
	ret := make([]Candidate[U], 0, n)
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

		ret = append(ret, Candidate[U]{Item: item.Item, Distance: item.Priority})
	}
	return ret, nil
}

func (bsp *bspTreeIndex[T, U]) SearchChannel(ctx context.Context, query []T) <-chan Candidate[U] {
	outputStream := make(chan Candidate[U], streamBufferSize)
	go func() error {
		defer close(outputStream)

		if !bsp.HasIndex() {
			bsp.Build(ctx)
		}

		env := linalg.NewLinAlgFromContext[T](ctx)

		queue := collection.NewPriorityQueue[queueItem](queueCapcitySize)
		for i := range bsp.Roots {
			queue.Push(queueItem{RootIdx: uint(i), NodeIdx: 0}, -math.MaxFloat64)
		}

		for 0 < queue.Len() {
			nodeWithPriority, err := queue.PopWithPriority()
			if err != nil {
				continue
			}

			ri := nodeWithPriority.Item.RootIdx
			root := bsp.Roots[ri]
			node := root.Nodes[nodeWithPriority.Item.NodeIdx]
			if node.Left == 0 && node.Right == 0 {
				for i := node.Begin; i < node.End; i++ {
					elem := bsp.Elements[root.Indice[i]]
					distance := float64(env.SqL2(query, elem.Feature))
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
			sqDistanceToCutPlane := node.CutPlane.Distance(query, env)
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

func (bsp bspTreeIndex[T, U]) HasIndex() bool {
	return 0 < len(bsp.Roots)
}

func (bsp *bspTreeIndex[T, U]) ClearIndex() {
	bsp.Roots = nil
}

func (bsp bspTreeIndex[T, U]) Empty() bool {
	return len(bsp.Elements) == 0
}

func (bsp bspTreeIndex[T, U]) Save(w io.Writer) error {
	return saveIndex(bsp, w)
}

func NewTreeIndex[T linalg.Number, U comparable](config TreeConfig, cpf CutPlaneFactory[T, U]) (*bspTreeIndex[T, U], error) {
	gob.Register(cpf.Default())

	if config.Dim == uint(0) {
		return nil, errors.New("dimension is not set")
	}

	leafs := config.Leafs
	if leafs == 0 {
		leafs = defaultLeafs
		log.Println("Leafs in given Config is not set. use default value", leafs)
	}

	trees := config.Trees
	if trees == 0 {
		trees = 1
		log.Println("Trees in given Config is not set. use default value", trees)
	}

	procs := config.Procs
	if procs == 0 {
		procs = uint(runtime.NumCPU())
		log.Println("Procs in given Config is not set. use default value", procs)
	}

	return &bspTreeIndex[T, U]{
		Elements: make([]TreeElement[T, U], 0),
		cpf:      cpf,
		Dim:      config.Dim,
		Leafs:    leafs,
		Trees:    trees,
		Procs:    procs,
	}, nil
}

func LoadTreeIndex[T linalg.Number, U comparable](r io.Reader, cpf CutPlaneFactory[T, U]) (*bspTreeIndex[T, U], error) {
	gob.Register(cpf.Default())

	index, err := loadIndex[bspTreeIndex[T, U]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}
