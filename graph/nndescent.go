package graph

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/sourcegraph/conc/pool"
)

type nndescentNode struct {
	Neighbors []uint
	Dists     []float32
	base      int
	accepted  int

	lastLowerBound float32
	lastAccepted   int
}

func (bgn *nndescentNode) Len() int {
	return len(bgn.Neighbors) - bgn.base
}

func (n *nndescentNode) Swap(i, j int) {
	n.Neighbors[i], n.Neighbors[j] = n.Neighbors[j], n.Neighbors[i]
	n.Dists[i], n.Dists[j] = n.Dists[j], n.Dists[i]
}

func (n *nndescentNode) Less(i, j int) bool {
	return n.Dists[i] < n.Dists[j]
}

func (bgn *nndescentNode) Add(idx uint, dist float32) {
	bgn.Neighbors = append(bgn.Neighbors, idx)
	bgn.Dists = append(bgn.Dists, dist)
}

func (bgn *nndescentNode) Shrink() bool {
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

func (bgn *nndescentNode) Peek() (uint, float32, bool) {
	if bgn.Len() == 0 {
		return 0, float32(math.Inf(0)), false
	}

	head := len(bgn.Neighbors) - 1
	return bgn.Neighbors[head], bgn.Dists[head], true
}

func (bgn *nndescentNode) Accept() bool {
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

func (bgn *nndescentNode) Drop() bool {
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
func (bgn *nndescentNode) Heapify() {
	// heapify
	n := bgn.Len()
	for i := n/2 - 1; i >= 0; i-- {
		i := len(bgn.Neighbors) - i - 1
		bgn.down(i, bgn.base)
	}
}

func (bgn *nndescentNode) getLesserSibling(lIdx, n int) int {
	ret := lIdx // left child
	if rIdx := lIdx - 1; n < rIdx && bgn.Less(rIdx, lIdx) {
		ret = rIdx // = 2*i + 2  // right child
	}

	return ret
}

// derived from container/heap
func (bgn *nndescentNode) down(i0, n int) bool {
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

func (n *nndescentNode) Merge(other nndescentNode) {
	for i := range other.Neighbors {
		n.Add(other.Neighbors[i], other.Dists[i])
	}
}

func (n *nndescentNode) Split(rho float64) nndescentNode {
	k := uint(rho * float64(n.Len()))
	rand.Shuffle(n.Len(), func(i, j int) {
		n.Neighbors[i], n.Neighbors[j] = n.Neighbors[j], n.Neighbors[i]
		n.Dists[i], n.Dists[j] = n.Dists[j], n.Dists[i]
	})

	splitted := nndescentNode{
		Neighbors: n.Neighbors[:k],
		Dists:     n.Dists[:k],
	}
	n.Neighbors = n.Neighbors[k:]
	n.Dists = n.Dists[k:]
	return splitted
}

type nndescentGraph struct {
	Nodes []nndescentNode
}

func newNndescentGraph(n uint) nndescentGraph {
	return nndescentGraph{
		Nodes: make([]nndescentNode, n),
	}
}

func newNndescentGraphFrom(g Graph, distFunc func(i, j uint) float32) nndescentGraph {
	ng := newNndescentGraph(uint(len(g.Nodes)))
	ng.traverse(func(i int, node *nndescentNode) {
		k := len(g.Nodes[i].Neighbors)
		node.Neighbors = make([]uint, 0, k)
		node.Dists = make([]float32, 0, k)

		for _, j := range g.Nodes[i].Neighbors {
			dist := distFunc(uint(i), j)
			node.Add(j, dist)
		}
	})
	return ng
}

func (g *nndescentGraph) traverse(f func(i int, node *nndescentNode)) {
	for i := range g.Nodes {
		f(i, &g.Nodes[i])
	}
}

func (g *nndescentGraph) Add(i, j uint, dist float32) {
	g.Nodes[i].Add(j, dist)
}

func (g nndescentGraph) Reverse(rho float64) nndescentGraph {
	rev := nndescentGraph{
		Nodes: make([]nndescentNode, len(g.Nodes)),
	}
	rev.traverse(func(i int, node *nndescentNode) {
		node.Neighbors = make([]uint, 0)
		node.Dists = make([]float32, 0)
	})
	g.traverse(func(i int, node *nndescentNode) {
		for _, ni := range node.Neighbors {
			rev.Nodes[ni].Add(uint(i), 0.0)
		}
	})

	return rev.Split(rho)
}

func (g *nndescentGraph) Merge(other nndescentGraph) error {
	if len(g.Nodes) != len(other.Nodes) {
		return fmt.Errorf("graph size mismatch: %d != %d", len(g.Nodes), len(other.Nodes))
	}

	g.traverse(func(i int, node *nndescentNode) {
		node.Merge(other.Nodes[i])
	})

	return nil
}

func (g *nndescentGraph) Split(rho float64) nndescentGraph {
	splitted := nndescentGraph{Nodes: make([]nndescentNode, len(g.Nodes))}
	splitted.traverse(func(i int, node *nndescentNode) {
		*node = g.Nodes[i].Split(rho)
	})

	return splitted
}

func (g *nndescentGraph) Join(i, j uint, dist float32) {
	g.Nodes[i].Add(j, dist)
	g.Nodes[j].Add(i, dist)
}

type Nndescent struct {
	fixed         nndescentGraph
	candidate     nndescentGraph
	k             uint
	rho           float64
	distFunc      func(i, j uint) float32
	maxGoroutines int
}

type Option func(*Nndescent)

func WithMaxGoroutines(maxGoroutines uint) Option {
	return func(n *Nndescent) {
		n.maxGoroutines = int(maxGoroutines)
	}
}

func NewNndescent(initGraph Graph, k uint, rho float64, f func(i, j uint) float32, options ...Option) Nndescent {
	n := uint(len(initGraph.Nodes))
	fixed := newNndescentGraph(n)
	candidate := newNndescentGraphFrom(initGraph, f)

	nndescent := Nndescent{
		fixed:         fixed,
		candidate:     candidate,
		k:             k,
		rho:           rho,
		distFunc:      f,
		maxGoroutines: runtime.NumCPU(),
	}
	for _, option := range options {
		option(&nndescent)
	}

	return nndescent
}

func (n *Nndescent) Update() uint {
	n.localJoin()
	return uint(n.prune())
}

func (n Nndescent) Create() Graph {
	ret := Graph{Nodes: make([]Node, len(n.fixed.Nodes))}
	for i := range ret.Nodes {
		ret.Nodes[i].Neighbors = make([]uint, 0, n.k)
		ret.Nodes[i].Neighbors = append(ret.Nodes[i].Neighbors, n.fixed.Nodes[i].Neighbors...)
		ret.Nodes[i].Neighbors = append(ret.Nodes[i].Neighbors, n.candidate.Nodes[i].Neighbors...)
	}
	return ret
}

func (n *Nndescent) localJoin() {
	locks := make([]sync.Mutex, len(n.candidate.Nodes))
	joinNodes := func(i, j uint, dist float32) {
		locks[i].Lock()
		n.candidate.Nodes[i].Add(j, dist)
		locks[i].Unlock()
		locks[j].Lock()
		n.candidate.Nodes[j].Add(i, dist)
		locks[j].Unlock()
	}

	old := n.fixed
	new := n.candidate.Split(n.rho)
	rold := old.Reverse(n.rho)
	rnew := new.Reverse(n.rho)

	p := pool.New().WithMaxGoroutines(n.maxGoroutines)
	for v := range n.candidate.Nodes {
		v := v
		p.Go(func() {
			for _, u1 := range new.Nodes[v].Neighbors {
				for _, u2 := range new.Nodes[v].Neighbors {
					if u2 <= u1 {
						continue
					}

					dist := n.distFunc(u1, u2)
					joinNodes(u1, u2, dist)
				}

				for _, u2 := range rnew.Nodes[v].Neighbors {
					if u2 <= u1 {
						continue
					}

					dist := n.distFunc(u1, u2)
					joinNodes(u1, u2, dist)
				}

				for _, u2 := range old.Nodes[v].Neighbors {
					if u2 == u1 {
						continue
					}

					dist := n.distFunc(u1, u2)
					joinNodes(u1, u2, dist)
				}

				for _, u2 := range rold.Nodes[v].Neighbors {
					if u2 == u1 {
						continue
					}

					dist := n.distFunc(u1, u2)
					joinNodes(u1, u2, dist)
				}
			}
		})
	}
	p.Wait()

	n.fixed.Merge(new)
}

func (n *Nndescent) prune() uint64 {
	changes := uint64(0)

	p := pool.New().WithMaxGoroutines(n.maxGoroutines)
	for v := range n.candidate.Nodes {
		v := v
		p.Go(func() {
			n.fixed.Nodes[v].Heapify()
			n.candidate.Nodes[v].Heapify()

			founds := make(map[uint]struct{})
			for i := 0; i < int(n.k); i++ {
				fi, fixedDist, fixedOk := n.fixed.Nodes[v].dropDuplicates(founds)
				ci, candDist, candOk := n.candidate.Nodes[v].dropDuplicates(founds)

				if !fixedOk && !candOk {
					break
				}

				if fixedDist <= candDist {
					n.fixed.Nodes[v].Accept()
					founds[fi] = struct{}{}
				} else {
					n.candidate.Nodes[v].Accept()
					founds[ci] = struct{}{}
				}
			}

			isFixedChanged := n.fixed.Nodes[v].Shrink()
			isCandidateChanged := n.candidate.Nodes[v].Shrink()
			if isFixedChanged || isCandidateChanged {
				atomic.AddUint64(&changes, 1)
			}
		})
	}
	p.Wait()

	return changes
}

func (bgn *nndescentNode) dropDuplicates(founds map[uint]struct{}) (uint, float32, bool) {
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
