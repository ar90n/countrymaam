package graph

import "testing"

func Test_ConvertToUndirected(t *testing.T) {
	g := Graph{
		Nodes: []Node{
			{Neighbors: []uint{1, 2, 3}},
			{Neighbors: []uint{0, 2}},
			{Neighbors: []uint{0, 1, 3}},
			{Neighbors: []uint{0, 4}},
			{Neighbors: []uint{2, 5, 6}},
			{Neighbors: []uint{}},
			{Neighbors: []uint{3, 4, 5}},
		},
	}

	g = ConvertToUndirected(g)

	for i := range g.Nodes {
		for _, j := range g.Nodes[i].Neighbors {
			found := false
			for _, k := range g.Nodes[j].Neighbors {
				if k == uint(i) {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("node %d is not neighbor of node %d", i, j)
			}
		}
	}
}
