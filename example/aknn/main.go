package main

import (
	"bytes"
	"context"
	"fmt"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/example"
	"github.com/ar90n/countrymaam/graph"
	"github.com/ar90n/countrymaam/index"
)

func main() {
	dim := uint(64)
	graphBuilder := graph.NewAKnnGraphBuilder[uint8]()
	graphBuilder.SetK(10).SetRho(1.0)
	builder := index.NewGraphIndexBuilder[uint8](dim, graphBuilder)

	features, err := example.ReadFeatures(dim)
	if err != nil {
		panic(err)
	}

	ctx := context.Background()
	ind, err := builder.Build(ctx, features)
	if err != nil {
		panic(err)
	}

	buf := make([]byte, 0)
	byteBuffer := bytes.NewBuffer(buf)
	if err := ind.Save(byteBuffer); err != nil {
		panic(err)
	}

	ind2, err := index.LoadGraphIndex[uint8](byteBuffer)
	if err != nil {
		panic(err)
	}

	query := []uint8{
		177, 73, 110, 135, 85, 153, 143, 73, 210, 208, 148, 50, 39, 165, 51, 201, 47, 102, 198, 55, 192, 42, 89, 189, 104, 86, 183, 162, 60, 145, 122, 104, 133, 200, 167, 51, 147, 167, 191, 220, 85, 75, 57, 72, 43, 150, 155, 53, 163, 171, 106, 115, 99, 78, 88, 48, 81, 214, 114, 126, 196, 214, 220, 75,
	}
	ch := ind2.SearchChannel(ctx, query)
	neighbors, err := countrymaam.Search(ch, 5, 32)
	if err != nil {
		panic(err)
	}

	for i, n := range neighbors {
		fmt.Printf("%d: %d, %f\n", i, n.Index, n.Distance)
	}
}
