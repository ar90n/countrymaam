package main

import (
	"bytes"
	"context"
	"fmt"

	"github.com/ar90n/countrymaam/graph"
	"github.com/ar90n/countrymaam/index"
)

func main() {
	dim := uint(64)
	graphBuilder := graph.NewAKnnGraphBuilder[uint8]()
	graphBuilder.SetK(30).SetRho(1.0)
	builder := index.NewGraphIndexBuilder[uint8, int](dim, graphBuilder)

	features, err := readFeatures(dim)
	if err != nil {
		panic(err)
	}

	items := make([]int, len(features))
	for i := range items {
		items[i] = i
	}

	ctx := context.Background()
	ind, err := builder.Build(ctx, features, items)
	if err != nil {
		panic(err)
	}

	buf := make([]byte, 0)
	byteBuffer := bytes.NewBuffer(buf)
	if err := ind.Save(byteBuffer); err != nil {
		panic(err)
	}

	ind2, err := index.LoadGraphIndex[uint8, int](byteBuffer)
	if err != nil {
		panic(err)
	}

	query := []uint8{
		177, 73, 110, 135, 85, 153, 143, 73, 210, 208, 148, 50, 39, 165, 51, 201, 47, 102, 198, 55, 192, 42, 89, 189, 104, 86, 183, 162, 60, 145, 122, 104, 133, 200, 167, 51, 147, 167, 191, 220, 85, 75, 57, 72, 43, 150, 155, 53, 163, 171, 106, 115, 99, 78, 88, 48, 81, 214, 114, 126, 196, 214, 220, 75,
	}
	neighbors, err := ind2.Search(ctx, query, 5, 32)
	if err != nil {
		panic(err)
	}

	for i, n := range neighbors {
		fmt.Printf("%d: %d, %f\n", i, n.Item, n.Distance)
	}
}
