package index

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/ar90n/countrymaam/linalg"
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
}

func (bgn builderGraphNode) Len() int {
	return len(bgn.Neighbors) - bgn.base
}

func (n *builderGraphNode) Swap(i, j int) {
	n.Neighbors[i], n.Neighbors[j] = n.Neighbors[j], n.Neighbors[i]
	n.Dists[i], n.Dists[j] = n.Dists[j], n.Dists[i]
}

func (n builderGraphNode) Less(i, j int) bool {
	return n.Dists[i] < n.Dists[j]
}

func (bgn *builderGraphNode) Add(idx uint, dist float32) {
	bgn.Neighbors = append(bgn.Neighbors, idx)
	bgn.Dists = append(bgn.Dists, dist)
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
		node.Neighbors = make([]uint, 0, len(g.Nodes[i].Neighbors))
		node.Dists = make([]float32, 0, len(g.Nodes[i].Neighbors))
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
}

func NewKnnGraphBuilder[T linalg.Number, U comparable](elements []TreeElement[T, U], k uint, rho float64) KnnGraphBuilder[T, U] {
	builder := KnnGraphBuilder[T, U]{
		Elements: elements,
		K:        k,
		Rho:      rho,
	}
	return builder
}

func (gb *KnnGraphBuilder[T, U]) Init(env linalg.Env[T]) error {
	nodes := make([]builderGraphNode, len(gb.Elements))
	for i := range nodes {
		i := uint(i)
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
	}

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
	isChanged := false
	for v := range gb.Candidate.Nodes {
		gb.Fixed.Nodes[v].Heapify()
		gb.Candidate.Nodes[v].Heapify()

		founds := make(map[uint]struct{})
		for i := 0; i < int(gb.K); i++ {
			fi, fixedDist, fixedOk := gb.Fixed.Nodes[v].dropDuplicates(founds)
			ci, candDist, candOk := gb.Candidate.Nodes[v].dropDuplicates(founds)

			if !fixedOk && !candOk {
				break
			}

			if fixedDist < candDist {
				gb.Fixed.Nodes[v].Accept()
				founds[fi] = struct{}{}
			} else {
				gb.Candidate.Nodes[v].Accept()
				founds[ci] = struct{}{}
			}
		}

		isChanged = gb.Fixed.Nodes[v].Shrink() || isChanged
		isChanged = gb.Candidate.Nodes[v].Shrink() || isChanged
	}

	return isChanged, nil
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

	for v := range gb.Candidate.Nodes {
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
	}

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
