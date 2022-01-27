# countrymaam
countrymaam is a simple implementation of similar vector search with pure Go.
This library is inspired by Flann, Annoy, and Faiss.

## Features
* Flat search index (`FlatIndex`)
* Kd-Tree base index (`KdTreeIndex` and `RandomizedKdTreeIndex`)
* Random-Projection Tree base index (`RpTreeIndex` And `RandomizedRpTreeIndex`)
* Serialize/Deserialize with gop

## Installation
```
$ go get github.com/ar90n/countrymaam
```

## How to use

### Sample code
```go
package main

import (
	"bytes"
	_ "embed"
	"encoding/csv"
	"fmt"
	"io"
	"strconv"

	"github.com/ar90n/countrymaam"
)

//dim064.csv is derived from http://cs.joensuu.fi/sipu/datasets/
//go:embed dim064.csv
var dim64 []byte

func readFeatures(dim uint) ([][]int, error) {
	r := csv.NewReader(bytes.NewReader(dim64))

	ret := make([][]int, 0)
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		feature := make([]int, dim)
		for i, text := range record {
			val, err := strconv.ParseInt(text, 10, 32)
			if err != nil {
				return nil, err
			}
			feature[i] = int(val)
		}
		ret = append(ret, feature)
	}

	return ret, nil
}

func main() {
	dim := uint(64)
	index := countrymaam.NewKdTreeIndex[int, int](dim, 8)

	features, err := readFeatures(dim)
	if err != nil {
		panic(err)
	}

	for i, f := range features {
		index.Add(f, i)
	}
	index.Build()

	query := []int{
		177, 73, 110, 135, 85, 153, 143, 73, 210, 208, 148, 50, 39, 165, 51, 201, 47, 102, 198, 55, 192, 42, 89, 189, 104, 86, 183, 162, 60, 145, 122, 104, 133, 200, 167, 51, 147, 167, 191, 220, 85, 75, 57, 72, 43, 150, 155, 53, 163, 171, 106, 115, 99, 78, 88, 48, 81, 214, 114, 126, 196, 214, 220, 75,
	}
	neighbors, err := index.Search(query, 5, 32)
	if err != nil {
		panic(err)
	}

	for i, n := range neighbors {
		fmt.Printf("%d: %d, %f\n", i, n.Item, n.Distance)
	}
}
```
### Result
```
0: 1023, 0.000000
1: 974, 6.000000
2: 992, 7.000000
3: 1001, 9.000000
4: 975, 9.000000
```


## Benchmark

### How to run
```bash
$ cd benchmark
$ make benchmark
$ ls ann-benchmarks/results/fashion-mnist-784-euclidean.png 
ann-benchmarks/results/fashion-mnist-784-euclidean.png
```

### Result

## See also
* [flann](https://github.com/flann-lib/flann)
* [annoy](https://github.com/spotify/annoy)
* [faiss](https://github.com/facebookresearch/faiss)
* [ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
* [Clustering basic benchmark](http://cs.joensuu.fi/sipu/datasets/)

## License
The source code is licensed MIT. The website content is licensed CC BY 4.0,see LICENSE.