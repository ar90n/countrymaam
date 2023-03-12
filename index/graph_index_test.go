package index

import (
	"fmt"
	"math"
	"sort"
	"testing"

	"github.com/ar90n/countrymaam/linalg"
	"github.com/stretchr/testify/assert"
)

//func Test_RandomizedKnn(t *testing.T) {
//	n := 8
//	d := 8
//	k := 2
//	elements := make([]TreeElement[float32, int], n)
//	for i := range elements {
//		elements[i].Feature = make([]float32, d)
//		for j := range elements[i].Feature {
//			elements[i].Feature[j] = rand.Float32()
//		}
//		elements[i].Item = i
//	}
//
//	env := linalg.NewLinAlg[float32](linalg.Config{})
//	knn, err := BuildRandomizedKnnGraph(elements, uint(k), env)
//	if err != nil {
//		t.Fatal(err)
//	}
//
//	g := graphviz.New()
//	graph, err := g.Graph()
//	if err != nil {
//		log.Fatal(err)
//	}
//	defer func() {
//		if err := graph.Close(); err != nil {
//			log.Fatal(err)
//		}
//		g.Close()
//	}()
//
//	gn := make([]*cgraph.Node, n)
//	for i := range knn.Nodes {
//		gn[i], err = graph.CreateNode(fmt.Sprintf("n%d", i))
//		if err != nil {
//			log.Fatal(err)
//		}
//	}
//
//	ge := make([]*cgraph.Edge, n*k)
//	for i := range knn.Nodes {
//		for j := range knn.Nodes[i].Neighbors {
//			ge[i*k+j], err = graph.CreateEdge(fmt.Sprintf("e%d", i*k+j), gn[i], gn[knn.Nodes[i].Neighbors[j].Index])
//			if err != nil {
//				log.Fatal(err)
//			}
//			ge[i*k+j].SetLabel(fmt.Sprintf("%f", knn.Nodes[i].Neighbors[j].Attr))
//		}
//	}
//	var buf bytes.Buffer
//
//	if err := g.Render(graph, graphviz.PNG, &buf); err != nil {
//		log.Fatal(err)
//	}
//
//	if err := g.RenderFilename(graph, graphviz.PNG, "./graph.png"); err != nil {
//		log.Fatal(err)
//	}
//}

func Test_Knn(t *testing.T) {
	v := [][]float32{{0.9382979, 0.02068228},
		{0.73769548, 0.27789461},
		{0.52404968, 0.66918405},
		{0.71130657, 0.04397154},
		{0.30150448, 0.99551993},
		{0.71053094, 0.80725171},
		{0.83579555, 0.27047663},
		{0.92257152, 0.35443522},
		{0.75475991, 0.03915375},
		{0.47519988, 0.79546934},
		{0.41285849, 0.91768804},
		{0.95689047, 0.53087249},
		{0.54369358, 0.72449079},
		{0.21832251, 0.95516216},
		{0.93584569, 0.75276496},
		{0.55507164, 0.35825514},
		{0.53575104, 0.31743178},
		{0.86958985, 0.79659692},
		{0.71037628, 0.12494913},
		{0.47549219, 0.91082355},
		{0.76717885, 0.70570274},
		{0.25268384, 0.49687757},
		{0.36881297, 0.00942773},
		{0.07258602, 0.26554888},
		{0.29408366, 0.89540884},
		{0.24222268, 0.3205058},
		{0.47095961, 0.57133958},
		{0.79535941, 0.37627325},
		{0.16554462, 0.10079731},
		{0.704429, 0.05787501},
		{0.80916261, 0.22355085},
		{0.6309418, 0.51406197}}

	n := len(v)
	k := 5
	elements := make([]TreeElement[float32, int], n)
	for i := range elements {
		elements[i].Feature = make([]float32, len(v[i]))
		for j := range elements[i].Feature {
			elements[i].Feature[j] = v[i][j]
		}
		elements[i].Item = i
	}

	env := linalg.NewLinAlg[float32](linalg.Config{})
	knn, err := BuildAknnGraph(elements, uint(k), env)
	if err != nil {
		t.Fatal(err)
	}

	ss := float32(0.0)
	fmt.Println(knn.Nodes)
	for i := range knn.Nodes {
		cs := float32(0.0)
		if len(knn.Nodes[i].Neighbors) != k {
			t.Fatal("wrong number of neighbors")
		}
		for j := range knn.Nodes[i].Neighbors {
			cs += float32(math.Sqrt(float64(env.SqL2(elements[i].Feature, elements[knn.Nodes[i].Neighbors[j]].Feature))))
			//cs += float32(math.Sqrt(float64(knn.Nodes[i].Neighbors[j].Attr)))
		}
		ss += cs
		fmt.Println(i, cs)
	}
	fmt.Println(ss)
	assert.InDelta(t, 28.686062, ss, 0.0001)

	// g := graphviz.New()
	// graph, err := g.Graph()
	//
	//	if err != nil {
	//		log.Fatal(err)
	//	}
	//
	//	defer func() {
	//		if err := graph.Close(); err != nil {
	//			log.Fatal(err)
	//		}
	//		g.Close()
	//	}()
	//
	// gn := make([]*cgraph.Node, n)
	//
	//	for i := range knn.Nodes {
	//		gn[i], err = graph.CreateNode(fmt.Sprintf("n%d", i))
	//		if err != nil {
	//			log.Fatal(err)
	//		}
	//	}
	//
	// ge := make([]*cgraph.Edge, n*k)
	//
	//	for i := range knn.Nodes {
	//		for j := range knn.Nodes[i].Neighbors {
	//			ge[i*k+j], err = graph.CreateEdge(fmt.Sprintf("e%d", i*k+j), gn[i], gn[knn.Nodes[i].Neighbors[j].Index])
	//			if err != nil {
	//				log.Fatal(err)
	//			}
	//			ge[i*k+j].SetLabel(fmt.Sprintf("%f", knn.Nodes[i].Neighbors[j].Attr))
	//		}
	//	}
	//
	// var buf bytes.Buffer
	//
	//	if err := g.Render(graph, graphviz.PNG, &buf); err != nil {
	//		log.Fatal(err)
	//	}
	//
	//	if err := g.RenderFilename(graph, graphviz.PNG, "./graph.png"); err != nil {
	//		log.Fatal(err)
	//	}
}

func Test_BuilderGraphNode(t *testing.T) {
	bgn := builderGraphNode{
		Neighbors: []uint{},
		Dists:     []float32{},
	}

	_, _, ok := bgn.Peek()
	assert.False(t, ok)

	bgn.Add(0, 0.3)
	bgn.Add(2, 0.7)
	bgn.Add(1, 0.1)
	bgn.Add(1, 0.1)
	bgn.Add(4, 0.01)
	bgn.Add(3, 0.4)

	assert.Len(t, bgn.Neighbors, 6)
	assert.Len(t, bgn.Dists, 6)

	idx, dist, ok := bgn.Peek()
	assert.Equal(t, uint(3), idx)
	assert.Equal(t, float32(0.4), dist)
	assert.True(t, ok)

	bgn.Swap(0, 5)
	idx, dist, ok = bgn.Peek()
	assert.Equal(t, uint(0), idx)
	assert.Equal(t, float32(0.3), dist)
	assert.True(t, ok)

	sort.Sort(&bgn)
	idx, dist, ok = bgn.Peek()
	assert.Equal(t, uint(2), idx)
	assert.Equal(t, float32(0.7), dist)
	assert.True(t, ok)

	bgn.Heapify()
	idx, dist, ok = bgn.Peek()
	assert.Equal(t, uint(4), idx)
	assert.Equal(t, float32(0.01), dist)
	assert.True(t, ok)

	bgn.Accept()
	idx, dist, ok = bgn.Peek()
	assert.Equal(t, uint(1), idx)
	assert.Equal(t, float32(0.1), dist)
	assert.True(t, ok)

	bgn.Accept()
	idx, dist, ok = bgn.Peek()
	assert.Equal(t, uint(1), idx)
	assert.Equal(t, float32(0.1), dist)
	assert.True(t, ok)

	bgn.Drop()
	idx, dist, ok = bgn.Peek()
	assert.Equal(t, uint(0), idx)
	assert.Equal(t, float32(0.3), dist)
	assert.True(t, ok)

	bgn.Accept()
	idx, dist, ok = bgn.Peek()
	assert.Equal(t, uint(3), idx)
	assert.Equal(t, float32(0.4), dist)
	assert.Equal(t, 2, bgn.Len())
	assert.True(t, ok)

	bgn.Drop()
	assert.Equal(t, 1, bgn.Len())
	fmt.Println(bgn.Neighbors, bgn.base, bgn.accepted)

	bgn.Shrink()

	assert.Equal(t, 3, bgn.Len())
	assert.Equal(t, []uint{4, 1, 0}, bgn.Neighbors)
}

func Test_BuilderGraphNode2(t *testing.T) {
	bgn := builderGraphNode{
		Neighbors: []uint{},
		Dists:     []float32{},
	}

	_, _, ok := bgn.Peek()
	assert.False(t, ok)

	bgn.Add(0, 0.3)
	bgn.Add(2, 0.7)
	bgn.Add(1, 0.1)
	bgn.Add(1, 0.1)
	bgn.Add(4, 0.01)
	bgn.Add(3, 0.4)

	bgn.Heapify()
	fmt.Println(bgn.Neighbors, bgn.base, bgn.accepted)

	bgn.Drop()
	fmt.Println(bgn.Neighbors, bgn.base, bgn.accepted)
	bgn.Drop()
	fmt.Println(bgn.Neighbors, bgn.base, bgn.accepted)
	bgn.Accept()
	fmt.Println(bgn.Neighbors, bgn.base, bgn.accepted)
	bgn.Drop()
	bgn.Accept()

	fmt.Println(bgn.Neighbors, bgn.base, bgn.accepted)
	bgn.Shrink()

	assert.Equal(t, 2, bgn.Len())
	assert.Equal(t, []uint{1, 3}, bgn.Neighbors)
}

func Test_BuilderGraphNode3(t *testing.T) {
	bgn := builderGraphNode{
		Neighbors: []uint{},
		Dists:     []float32{},
	}

	_, _, ok := bgn.Peek()
	assert.False(t, ok)

	bgn.Add(0, 0.3)
	bgn.Add(2, 0.7)

	bgn.Heapify()

	bgn.Drop()
	bgn.Drop()
	bgn.Drop()
	bgn.Shrink()

	assert.Equal(t, 0, bgn.Len())
}
