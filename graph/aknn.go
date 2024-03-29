package graph

import (
	"fmt"
	"math/rand"
	"runtime"

	"github.com/ar90n/countrymaam/linalg"
	"github.com/sourcegraph/conc/pool"
)

type AKnnGraphBuilder[T linalg.Number] struct {
	k          uint
	rho        float64
	maxIter    uint
	maxChanges uint
}

func NewAKnnGraphBuilder[T linalg.Number]() *AKnnGraphBuilder[T] {
	const defaultK = 15
	const defaultRho = 0.7
	const defaultMaxIter = 4096
	return &AKnnGraphBuilder[T]{k: defaultK, rho: defaultRho, maxIter: defaultMaxIter}
}

func (agc *AKnnGraphBuilder[T]) SetK(k uint) *AKnnGraphBuilder[T] {
	agc.k = k
	return agc
}

func (agc *AKnnGraphBuilder[T]) SetRho(rho float64) *AKnnGraphBuilder[T] {
	agc.rho = rho
	return agc
}

func (agc *AKnnGraphBuilder[T]) SetMaxIter(maxIter uint) *AKnnGraphBuilder[T] {
	agc.maxIter = maxIter
	return agc
}

func (agc *AKnnGraphBuilder[T]) SetMaxChanges(maxChanges uint) *AKnnGraphBuilder[T] {
	agc.maxChanges = maxChanges
	return agc
}

func (agc AKnnGraphBuilder[T]) GetPrameterString() string {
	return fmt.Sprintf("k=%d,rho=%f,maxIter=%d", agc.k, agc.rho, agc.maxIter)
}

func (agc *AKnnGraphBuilder[T]) Build(n uint, distFunc func(i, j uint) float32) (Graph, error) {
	rg := newRandomizedKnGraph(n, agc.k)
	nndescent := NewNndescent(rg, agc.k, agc.rho, distFunc)

	for i := uint(0); i < agc.maxIter; i++ {
		changes := nndescent.Update()
		if changes <= agc.maxChanges {
			break
		}
	}

	return nndescent.Create(), nil
}

func newRandomizedKnGraph(n, k uint) Graph {
	nodes := make([]Node, n)

	procs := uint(runtime.NumCPU())
	p := pool.New().WithMaxGoroutines(int(procs))
	for i := range nodes {
		i := uint(i)
		p.Go(func() {
			nodes[i].Neighbors = make([]uint, 0, k)

			ignores := map[uint]struct{}{
				i: {},
			}
			for uint(len(ignores)) <= k {
				idx := uint(rand.Int31n(int32(len(nodes))))
				if _, ok := ignores[idx]; ok {
					continue
				}
				ignores[idx] = struct{}{}

				nodes[i].Neighbors = append(nodes[i].Neighbors, idx)
			}
		})
	}
	p.Wait()

	return Graph{Nodes: nodes}
}
