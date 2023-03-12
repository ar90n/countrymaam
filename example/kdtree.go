package main

import (
	"context"
	_ "embed"
	"fmt"

	"github.com/ar90n/countrymaam/cut_plane"
	"github.com/ar90n/countrymaam/index"
)

func main() {
	dim := uint(64)
	ctx := context.Background()
	index, err := index.NewTreeIndex(
		index.TreeConfig{
			Dim:   dim,
			Leafs: 8,
			Trees: 8,
		},
		cut_plane.NewKdCutPlaneFactory[uint8, int](100, 5),
	)
	if err != nil {
		panic(err)
	}

	features, err := readFeatures(dim)
	if err != nil {
		panic(err)
	}

	for i, f := range features {
		index.Add(f, i)
	}
	index.Build(ctx)

	query := []uint8{
		177, 73, 110, 135, 85, 153, 143, 73, 210, 208, 148, 50, 39, 165, 51, 201, 47, 102, 198, 55, 192, 42, 89, 189, 104, 86, 183, 162, 60, 145, 122, 104, 133, 200, 167, 51, 147, 167, 191, 220, 85, 75, 57, 72, 43, 150, 155, 53, 163, 171, 106, 115, 99, 78, 88, 48, 81, 214, 114, 126, 196, 214, 220, 75,
	}
	neighbors, err := index.Search(ctx, query, 5, 32)
	if err != nil {
		panic(err)
	}

	for i, n := range neighbors {
		fmt.Printf("%d: %d, %f\n", i, n.Item, n.Distance)
	}
}
