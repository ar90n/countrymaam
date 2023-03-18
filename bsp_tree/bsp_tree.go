package bsp_tree

import (
	"encoding/gob"

	"github.com/ar90n/countrymaam/collection"
	"github.com/ar90n/countrymaam/linalg"
)

type BspTree[T linalg.Number] struct {
	Indice []int
	Nodes  []Node[T]
}

func (r *BspTree[T]) addNode(node Node[T]) uint {
	nc := uint(len(r.Nodes))
	r.Nodes = append(r.Nodes, node)

	return nc
}

func (r *BspTree[T]) buildSubTree(features [][]T, indice []int, leafs, offset uint, env linalg.Env[T], cf func(features [][]T, indice []int, env linalg.Env[T]) (CutPlane[T], error)) (uint, error) {
	ec := uint(len(indice))
	if ec == 0 {
		return 0, nil
	}

	curIdx := r.addNode(Node[T]{
		Begin: offset,
		End:   offset + ec,
	})

	if ec <= leafs {
		return curIdx, nil
	}

	cutPlane, err := cf(features, indice, env)
	if err != nil {
		return 0, err
	}
	r.Nodes[curIdx].CutPlane = cutPlane

	mid := collection.Partition(indice, func(i int) bool {
		return cutPlane.Evaluate(features[i], env)
	})

	left, err := r.buildSubTree(features, indice[:mid], leafs, offset, env, cf)
	if err != nil {
		return 0, err
	}
	r.Nodes[curIdx].Left = left

	right, err := r.buildSubTree(features, indice[mid:], leafs, offset+mid, env, cf)
	if err != nil {
		return 0, err
	}
	r.Nodes[curIdx].Right = right

	return curIdx, nil
}

type CutPlane[T linalg.Number] interface {
	Evaluate(feature []T, env linalg.Env[T]) bool
	Distance(feature []T, env linalg.Env[T]) float64
}

type Node[T linalg.Number] struct {
	CutPlane CutPlane[T]
	Begin    uint
	End      uint
	Left     uint
	Right    uint
}

type BspTreeBuilder[T linalg.Number] interface {
	Build(features [][]T, env linalg.Env[T]) (BspTree[T], error)
}

func Register[T linalg.Number]() {
	gob.Register(&kdCutPlane[T]{})
	gob.Register(&rpCutPlane[T]{})
}
