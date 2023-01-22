package main

import (
	"bytes"
	"context"
	_ "embed"
	"encoding/csv"
	"fmt"
	"io"
	"sort"
	"strconv"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/ar90n/countrymaam/pipeline"
)

//go:embed dim064.csv
var dim64 []byte

func readFeatures(dim uint) ([][]uint8, error) {
	r := csv.NewReader(bytes.NewReader(dim64))

	ret := make([][]uint8, 0)
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		feature := make([]uint8, dim)
		for i, text := range record {
			val, err := strconv.ParseInt(text, 10, 32)
			if err != nil {
				return nil, err
			}
			feature[i] = uint8(val)
		}
		ret = append(ret, feature)
	}

	return ret, nil
}

func main() {
	dim := uint(64)
	env := linalg.NewLinAlgUint8(linalg.LinAlgOptions{})
	ctx := context.Background()
	//index := countrymaam.NewKdTreeIndex[uint8, int](dim, 8, env)
	index := countrymaam.NewRandomizedKdTreeIndex[uint8, int](dim, 8, 8, env)
	//index := countrymaam.NewRandomizedRpTreeIndex[uint8, int](dim, 8, 8, env)
	//index := countrymaam.NewKdTreeIndex[uint8, int](dim, 8, env)

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

	ch := index.SearchChannel(ctx, query)
	ch = pipeline.Unique(ctx, ch)
	ch = pipeline.Take(ctx, 64, ch)
	results := pipeline.ToSlice(ctx, ch)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})

	fmt.Println()
	for i, n := range results[:5] {
		fmt.Printf("%d: %d, %f\n", i, n.Item, n.Distance)
	}

}
