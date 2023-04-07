package graph

import "github.com/ar90n/countrymaam/linalg"

type Graph struct {
	Nodes []Node
}

type Node struct {
	Neighbors []uint
}

type GraphBuilder interface {
	Build(n uint, distFunc func(i, j uint) float32) (Graph, error)
	GetPrameterString() string
}

func Register[T linalg.Number]() {
}

func ConvertToUndirected(g Graph) Graph {
	neighborSets := make([]map[uint]struct{}, len(g.Nodes))
	for i := range neighborSets {
		neighborSets[i] = make(map[uint]struct{})
	}

	for i := range g.Nodes {
		for _, j := range g.Nodes[i].Neighbors {
			neighborSets[i][j] = struct{}{}
			neighborSets[j][uint(i)] = struct{}{}
		}
	}

	ret := Graph{Nodes: make([]Node, len(g.Nodes))}
	for i := range ret.Nodes {
		ret.Nodes[i].Neighbors = make([]uint, 0, len(neighborSets[i]))
		for j := range neighborSets[i] {
			ret.Nodes[i].Neighbors = append(ret.Nodes[i].Neighbors, j)
		}
	}

	return ret
}
