package index

import (
	"context"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/sourcegraph/conc/pool"
)

type Graph struct {
	Nodes []GraphNode
}

type GraphNode struct {
	Neighbors []uint
}

type builderGraphNode struct {
	Neighbors []uint
	Dists     []float32
	base      int
	accepted  int

	lastLowerBound float32
	lastAccepted   int
	mu             sync.Mutex
}

func (bgn *builderGraphNode) Len() int {
	return len(bgn.Neighbors) - bgn.base
}

func (n *builderGraphNode) Swap(i, j int) {
	n.Neighbors[i], n.Neighbors[j] = n.Neighbors[j], n.Neighbors[i]
	n.Dists[i], n.Dists[j] = n.Dists[j], n.Dists[i]
}

func (n *builderGraphNode) Less(i, j int) bool {
	return n.Dists[i] < n.Dists[j]
}

func (bgn *builderGraphNode) Add(idx uint, dist float32) {
	bgn.mu.Lock()
	bgn.Neighbors = append(bgn.Neighbors, idx)
	bgn.Dists = append(bgn.Dists, dist)
	bgn.mu.Unlock()
}

func (bgn *builderGraphNode) Shrink() bool {
	isChanged := (bgn.lastAccepted != bgn.accepted)
	if 0 < bgn.accepted {
		isChanged = isChanged || bgn.lastLowerBound != bgn.Dists[bgn.accepted-1]
		bgn.lastLowerBound = bgn.Dists[bgn.accepted-1]
	}

	bgn.Neighbors = bgn.Neighbors[:bgn.accepted]
	bgn.Dists = bgn.Dists[:bgn.accepted]
	bgn.lastAccepted = bgn.accepted
	bgn.base = 0
	bgn.accepted = 0
	return isChanged
}

func (bgn *builderGraphNode) Peek() (uint, float32, bool) {
	if bgn.Len() == 0 {
		return 0, float32(math.Inf(0)), false
	}

	head := len(bgn.Neighbors) - 1
	return bgn.Neighbors[head], bgn.Dists[head], true
}

func (bgn *builderGraphNode) Accept() bool {
	if bgn.Len() == 0 {
		return false
	}

	head := len(bgn.Neighbors) - 1
	bgn.Swap(head, bgn.base)
	bgn.base++

	bgn.Swap(bgn.accepted, bgn.base-1)
	bgn.accepted++
	bgn.down(head, bgn.base)
	return true
}

func (bgn *builderGraphNode) Drop() bool {
	if bgn.Len() == 0 {
		return false
	}

	head := len(bgn.Neighbors) - 1
	bgn.Swap(head, bgn.base)
	bgn.base++

	bgn.down(head, bgn.base)
	return true
}

// derived from container/heap
func (bgn *builderGraphNode) Heapify() {
	// heapify
	n := bgn.Len()
	for i := n/2 - 1; i >= 0; i-- {
		i := len(bgn.Neighbors) - i - 1
		bgn.down(i, bgn.base)
	}
}

func (bgn builderGraphNode) getLesserSibling(lIdx, n int) int {
	ret := lIdx // left child
	if rIdx := lIdx - 1; n < rIdx && bgn.Less(rIdx, lIdx) {
		ret = rIdx // = 2*i + 2  // right child
	}

	return ret
}

// derived from container/heap
func (bgn *builderGraphNode) down(i0, n int) bool {
	n = n - 1
	head := len(bgn.Neighbors) - 1
	i := i0

	for {
		j1 := head + (2*(i-head) - 1)
		if j1 <= n || head < j1 { // head <= j1 after int overflow
			break
		}
		j := bgn.getLesserSibling(j1, n)

		if !bgn.Less(j, i) {
			break
		}
		bgn.Swap(i, j)
		i = j
	}
	return i < i0
}

func (n *builderGraphNode) Merge(other builderGraphNode) {
	for i := range other.Neighbors {
		n.Add(other.Neighbors[i], other.Dists[i])
	}
}

func (n *builderGraphNode) Split(rho float64) builderGraphNode {
	k := uint(rho * float64(n.Len()))
	rand.Shuffle(n.Len(), func(i, j int) {
		n.Neighbors[i], n.Neighbors[j] = n.Neighbors[j], n.Neighbors[i]
		n.Dists[i], n.Dists[j] = n.Dists[j], n.Dists[i]
	})

	splitted := builderGraphNode{
		Neighbors: n.Neighbors[:k],
		Dists:     n.Dists[:k],
	}
	n.Neighbors = n.Neighbors[k:]
	n.Dists = n.Dists[k:]
	return splitted
}

type builderGraph struct {
	Nodes []builderGraphNode
}

func (g *builderGraph) traverse(f func(i int, node *builderGraphNode)) {
	for i := range g.Nodes {
		f(i, &g.Nodes[i])
	}
}

func (g *builderGraph) Add(i, j uint, dist float32) {
	g.Nodes[i].Add(j, dist)
}

func (g builderGraph) Reverse(rho float64) builderGraph {
	rev := builderGraph{
		Nodes: make([]builderGraphNode, len(g.Nodes)),
	}
	rev.traverse(func(i int, node *builderGraphNode) {
		node.Neighbors = make([]uint, 0)
		node.Dists = make([]float32, 0)
	})
	g.traverse(func(i int, node *builderGraphNode) {
		for _, ni := range node.Neighbors {
			rev.Nodes[ni].Add(uint(i), 0.0)
		}
	})

	return rev.Split(rho)
}

func (g *builderGraph) Merge(other builderGraph) error {
	if len(g.Nodes) != len(other.Nodes) {
		return fmt.Errorf("graph size mismatch: %d != %d", len(g.Nodes), len(other.Nodes))
	}

	g.traverse(func(i int, node *builderGraphNode) {
		node.Merge(other.Nodes[i])
	})

	return nil
}

func (g *builderGraph) Split(rho float64) builderGraph {
	splitted := builderGraph{Nodes: make([]builderGraphNode, len(g.Nodes))}
	splitted.traverse(func(i int, node *builderGraphNode) {
		*node = g.Nodes[i].Split(rho)
	})

	return splitted
}

func (g builderGraph) ToGraph() Graph {
	graph := Graph{
		Nodes: make([]GraphNode, len(g.Nodes)),
	}

	g.traverse(func(i int, node *builderGraphNode) {
		graph.Nodes[i].Neighbors = make([]uint, len(node.Neighbors))
		copy(graph.Nodes[i].Neighbors, node.Neighbors)
	})

	return graph
}

type KnnGraphBuilder[T linalg.Number, U comparable] struct {
	Elements  []TreeElement[T, U]
	Fixed     builderGraph
	Candidate builderGraph
	K         uint
	Rho       float64
	Procs     uint
}

func NewKnnGraphBuilder[T linalg.Number, U comparable](elements []TreeElement[T, U], k uint, rho float64) KnnGraphBuilder[T, U] {
	procs := uint(runtime.NumCPU())
	builder := KnnGraphBuilder[T, U]{
		Elements: elements,
		K:        k,
		Rho:      rho,
		Procs:    procs,
	}
	return builder
}

func (gb *KnnGraphBuilder[T, U]) Init(env linalg.Env[T]) error {
	nodes := make([]builderGraphNode, len(gb.Elements))

	p := pool.New().WithMaxGoroutines(int(gb.Procs))
	for i := range nodes {
		i := uint(i)
		p.Go(func() {
			nodes[i].Neighbors = make([]uint, 0, gb.K)
			nodes[i].Dists = make([]float32, 0, gb.K)

			ignores := map[uint]struct{}{
				i: {},
			}
			for uint(len(ignores)) <= gb.K {
				idx := uint(rand.Int31n(int32(len(nodes))))
				if _, ok := ignores[idx]; ok {
					continue
				}
				ignores[idx] = struct{}{}

				dist := env.SqL2(gb.Elements[i].Feature, gb.Elements[idx].Feature)
				nodes[i].Add(idx, dist)
			}
		})
	}
	p.Wait()

	gb.Fixed = builderGraph{Nodes: make([]builderGraphNode, len(gb.Elements))}
	gb.Candidate = builderGraph{Nodes: nodes}
	return nil
}

func (gb *KnnGraphBuilder[T, U]) Update(env linalg.Env[T]) (bool, error) {
	gb.addCandidates(env)
	ret, err := gb.removeRedundat()
	if err != nil {
		return ret, err
	}

	return ret, nil
}

func (gb KnnGraphBuilder[T, U]) Build(k uint) Graph {
	ret := Graph{Nodes: make([]GraphNode, len(gb.Elements))}
	for i := range ret.Nodes {
		ret.Nodes[i].Neighbors = make([]uint, 0, k)
	}

	for i := range gb.Fixed.Nodes {
		for j := range gb.Fixed.Nodes[i].Neighbors {
			ret.Nodes[i].Neighbors = append(ret.Nodes[i].Neighbors, gb.Fixed.Nodes[i].Neighbors[j])
		}
	}

	for i := range gb.Candidate.Nodes {
		for j := range gb.Candidate.Nodes[i].Neighbors {
			ret.Nodes[i].Neighbors = append(ret.Nodes[i].Neighbors, gb.Candidate.Nodes[i].Neighbors[j])
		}
	}

	return ret
}

func (gb *KnnGraphBuilder[T, U]) removeRedundat() (bool, error) {
	changes := uint64(0)

	p := pool.New().WithMaxGoroutines(int(gb.Procs)).WithErrors()
	for v := range gb.Candidate.Nodes {
		v := v
		p.Go(func() error {
			gb.Fixed.Nodes[v].Heapify()
			gb.Candidate.Nodes[v].Heapify()

			founds := make(map[uint]struct{})
			for i := 0; i < int(gb.K); i++ {
				fi, fixedDist, fixedOk := gb.Fixed.Nodes[v].dropDuplicates(founds)
				ci, candDist, candOk := gb.Candidate.Nodes[v].dropDuplicates(founds)

				if !fixedOk && !candOk {
					break
				}

				if fixedDist <= candDist {
					gb.Fixed.Nodes[v].Accept()
					founds[fi] = struct{}{}
				} else {
					gb.Candidate.Nodes[v].Accept()
					founds[ci] = struct{}{}
				}
			}

			isFixedChanged := gb.Fixed.Nodes[v].Shrink()
			isCandidateChanged := gb.Candidate.Nodes[v].Shrink()
			if isFixedChanged || isCandidateChanged {
				atomic.AddUint64(&changes, 1)
			}

			return nil
		})
	}

	err := p.Wait()
	//isConverged := changes < 20
	isConverged := changes != 0
	return isConverged, err
}

func (gb *KnnGraphBuilder[T, U]) addRelationShip(i, j uint, env linalg.Env[T]) {
	dist := env.SqL2(gb.Elements[i].Feature, gb.Elements[j].Feature)
	gb.Candidate.Add(i, j, dist)
	gb.Candidate.Add(j, i, dist)
}

func (gb *KnnGraphBuilder[T, U]) addCandidates(env linalg.Env[T]) {
	old := gb.Fixed
	new := gb.Candidate.Split(gb.Rho)
	rold := old.Reverse(gb.Rho)
	rnew := new.Reverse(gb.Rho)

	p := pool.New().WithMaxGoroutines(int(gb.Procs)).WithErrors()
	for v := range gb.Candidate.Nodes {
		v := v
		p.Go(func() error {
			for _, u1 := range new.Nodes[v].Neighbors {
				for _, u2 := range new.Nodes[v].Neighbors {
					if u2 <= u1 {
						continue
					}

					gb.addRelationShip(u1, u2, env)
				}

				for _, u2 := range rnew.Nodes[v].Neighbors {
					if u2 <= u1 {
						continue
					}

					gb.addRelationShip(u1, u2, env)
				}

				for _, u2 := range old.Nodes[v].Neighbors {
					if u2 == u1 {
						continue
					}

					gb.addRelationShip(u1, u2, env)
				}

				for _, u2 := range rold.Nodes[v].Neighbors {
					if u2 == u1 {
						continue
					}

					gb.addRelationShip(u1, u2, env)
				}
			}

			return nil
		})
	}
	p.Wait()
	gb.Fixed.Merge(new)
}

func (bgn *builderGraphNode) dropDuplicates(founds map[uint]struct{}) (uint, float32, bool) {
	var idx uint
	var dist float32
	var retOk bool
	for {
		idx, dist, retOk = bgn.Peek()
		if _, ok := founds[idx]; !ok || !retOk {
			break
		}

		bgn.Drop()
	}

	return idx, dist, retOk
}

type AKnnGraphIndex[T linalg.Number, U comparable] struct {
	Elements []TreeElement[T, U]
	K        uint
	Rho      float64
	G        Graph
}

func (gi *AKnnGraphIndex[T, U]) Add(feature []T, item U) {
	gi.Elements = append(gi.Elements, TreeElement[T, U]{Item: item, Feature: feature})
}

func (gi AKnnGraphIndex[T, U]) Search(ctx context.Context, query []T, n uint, maxCandidates uint) ([]countrymaam.Candidate[U], error) {
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

func (gi AKnnGraphIndex[T, U]) findApproxNearest(entry uint, query []T, env linalg.Env[T]) collection.WithPriority[uint] {
	curIdx := entry
	curDist := float64(env.SqL2(query, gi.Elements[curIdx].Feature))

	q := collection.NewPriorityQueue[uint](0)
	q.Push(curIdx, curDist)

	best := collection.WithPriority[uint]{
		Item:     uint(0),
		Priority: float64(math.MaxFloat64),
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

			dist := float64(env.SqL2(query, gi.Elements[e].Feature))
			q.Push(e, dist)
		}
	}

	return best
}

func (gi AKnnGraphIndex[T, U]) SearchChannel(ctx context.Context, query []T) <-chan countrymaam.Candidate[U] {
	env := linalg.NewLinAlgFromContext[T](ctx)
	outputStream := make(chan countrymaam.Candidate[U], streamBufferSize)

	go func() {
		defer close(outputStream)

		approxNearest := gi.findApproxNearest(0, query, linalg.NewLinAlgFromContext[T](ctx))

		q := collection.NewPriorityQueue[uint](0)
		q.Push(approxNearest.Item, approxNearest.Priority)

		visited := map[uint]struct{}{approxNearest.Item: {}}
		for {
			cur, err := q.PopWithPriority()
			if err != nil {
				return
			}

			select {
			case <-ctx.Done():
				return
			case outputStream <- countrymaam.Candidate[U]{
				Item:     gi.Elements[cur.Item].Item,
				Distance: cur.Priority,
			}:
			}

			for _, e := range gi.G.Nodes[cur.Item].Neighbors {
				if _, found := visited[e]; found {
					continue
				}
				visited[e] = struct{}{}

				dist := float64(env.SqL2(query, gi.Elements[e].Feature))
				q.Push(e, dist)
			}
		}
	}()

	return outputStream
}

func (gi *AKnnGraphIndex[T, U]) Build(ctx context.Context) error {
	env := linalg.NewLinAlgFromContext[T](ctx)

	builder := NewKnnGraphBuilder(gi.Elements, gi.K, gi.Rho)
	err := builder.Init(env)
	if err != nil {
		return err
	}

	isConverged := false
	for !isConverged {
		isConverged, err = builder.Update(env)
		if err != nil {
			return err
		}
	}

	gi.G = builder.Build(gi.K)
	return nil
}

func (gi AKnnGraphIndex[T, U]) HasIndex() bool {
	return 0 < len(gi.G.Nodes)
}

func (gi AKnnGraphIndex[T, U]) Save(w io.Writer) error {
	return saveIndex(gi, w)
}

func BuildAknnGraph(elements []TreeElement[float32, int], k uint, env linalg.Env[float32]) (Graph, error) {
	builder := NewKnnGraphBuilder(elements, k, 1.0)
	err := builder.Init(env)
	if err != nil {
		return Graph{}, err
	}

	isConverged := true
	for isConverged {
		isConverged, err = builder.Update(env)
		if err != nil {
			return Graph{}, err
		}
	}

	return builder.Build(k), nil
}

func NewAKnnGraphIndex[T linalg.Number, U comparable](k uint, rho float64) *AKnnGraphIndex[T, U] {
	gob.Register(AKnnGraphIndex[T, U]{})

	return &AKnnGraphIndex[T, U]{
		K:   k,
		Rho: rho,
	}
}

func LoadAKnnIndex[T linalg.Number, U comparable](r io.Reader) (*AKnnGraphIndex[T, U], error) {
	gob.Register(AKnnGraphIndex[T, U]{})

	index, err := loadIndex[AKnnGraphIndex[T, U]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}
