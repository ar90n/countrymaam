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
}

func Register[T linalg.Number]() {
}
