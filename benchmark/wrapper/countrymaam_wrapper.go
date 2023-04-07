package main

// #include <stdlib.h>
import "C"

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime/pprof"
	"unsafe"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/bsp_tree"
	"github.com/ar90n/countrymaam/graph"
	"github.com/ar90n/countrymaam/index"
)

type Symbol = uint64

type algo struct {
	name        string
	index       countrymaam.Index[float32]
	paramString string
	useProfile  bool
}

var (
	indexes      = map[Symbol]*algo{}
	profiFileDir = "/tmp"
)

var idx = uint64(0)

func convToBool(b C._Bool) bool {
	return b == C._Bool(true)
}

func newSymbol(name string, index countrymaam.Index[float32], paramString string, useProfile bool) Symbol {
	//symbol := Symbol(uintptr(unsafe.Pointer(index)))
	//indexes[symbol] = index
	symbol := idx
	indexes[symbol] = &algo{
		name:        name,
		index:       index,
		paramString: paramString,
		useProfile:  useProfile,
	}
	idx += 1

	return symbol
}

func convertToSlice[T any, U any](arr *U, arrRows, arrCols C.int) [][]T {
	rows := int(arrRows)
	cols := int(arrCols)
	rawArr := (*[1 << 28]T)(unsafe.Pointer(arr))[:rows*cols]

	features := make([][]T, rows)
	for i := 0; i < rows; i++ {
		features[i] = rawArr[i*cols : (i+1)*cols]
	}
	return features
}

func getProfileFileName(name, task, paramString string) string {
	prefix := fmt.Sprintf("%s_%s", name, task)
	if paramString == "" {
		return fmt.Sprintf("%s.prof", prefix)
	}
	return fmt.Sprintf("%s_%s.prof", prefix, paramString)
}

func startProfiler(fileName string) func() {
	pprofFilePath := filepath.Join(profiFileDir, fileName)
	f, err := os.Create(pprofFilePath)
	if err != nil {
		panic(err)
	}

	if err := pprof.StartCPUProfile(f); err != nil {
		f.Close()
		panic(err)
	}

	return func() {
		pprof.StopCPUProfile()
		f.Close()
	}
}

//export NewFlatIndex
func NewFlatIndex(arr *C.float, arrRows, arrCols C.int, useProfile C._Bool) Symbol {
	features := convertToSlice[float32](arr, arrRows, arrCols)
	dim := uint(len(features[0]))

	builder := index.NewFlatIndexBuilder[float32](dim)

	paramString := builder.GetPrameterString()
	if convToBool(useProfile) {
		pprofFileName := getProfileFileName("flat", "train", paramString)
		callback := startProfiler(pprofFileName)
		defer callback()
	}

	ctx := context.Background()
	idx, err := builder.Build(ctx, features)
	if err != nil {
		panic(err)
	}

	return newSymbol("flat", idx, paramString, convToBool(useProfile))
}

//export NewKdTreeIndex
func NewKdTreeIndex(arr *C.float, arrRows, arrCols C.int, leafs, trees, sampleFeatures, topKCandidates C.int, useProfile C._Bool) Symbol {
	features := convertToSlice[float32](arr, arrRows, arrCols)
	dim := uint(len(features[0]))

	kdTreeBuilder := bsp_tree.NewKdTreeBuilder[float32]()
	if 0 < leafs {
		kdTreeBuilder.SetLeafs(uint(leafs))
	}
	if 0 < sampleFeatures {
		kdTreeBuilder.SetSampleFeatures(uint(sampleFeatures))
	}
	if 0 < topKCandidates {
		kdTreeBuilder.SetTopKCandidates(uint(topKCandidates))
	}

	builder := index.NewBspTreeIndexBuilder[float32](dim, kdTreeBuilder)
	if 0 < trees {
		builder.SetTrees(uint(trees))
	}

	paramString := builder.GetPrameterString()
	if convToBool(useProfile) {
		pprofFileName := getProfileFileName("kdtree", "train", paramString)
		callback := startProfiler(pprofFileName)
		defer callback()
	}

	ctx := context.Background()
	idx, err := builder.Build(ctx, features)
	if err != nil {
		panic(err)
	}

	return newSymbol("kdtree", idx, paramString, convToBool(useProfile))
}

//export NewRpTreeIndex
func NewRpTreeIndex(arr *C.float, arrRows, arrCols C.int, leafs, trees, sampleFeatures C.int, useProfile C._Bool) Symbol {
	features := convertToSlice[float32](arr, arrRows, arrCols)
	dim := uint(len(features[0]))

	rpTreeBuilder := bsp_tree.NewRpTreeBuilder[float32]()
	if 0 < leafs {
		rpTreeBuilder.SetLeafs(uint(leafs))
	}
	if 0 < sampleFeatures {
		rpTreeBuilder.SetSampleFeatures(uint(sampleFeatures))
	}

	builder := index.NewBspTreeIndexBuilder[float32](dim, rpTreeBuilder)
	if 0 < trees {
		builder.SetTrees(uint(trees))
	}

	paramString := builder.GetPrameterString()
	if convToBool(useProfile) {
		pprofFileName := getProfileFileName("rptree", "train", paramString)
		callback := startProfiler(pprofFileName)
		defer callback()
	}

	ctx := context.Background()
	idx, err := builder.Build(ctx, features)
	if err != nil {
		panic(err)
	}

	return newSymbol("rptree", idx, paramString, convToBool(useProfile))
}

//export NewAKnnIndex
func NewAKnnIndex(arr *C.float, arrRows, arrCols C.int, k C.int, rho C.float, useProfile C._Bool) Symbol {
	features := convertToSlice[float32](arr, arrRows, arrCols)
	dim := uint(len(features[0]))

	graphBuilder := graph.NewAKnnGraphBuilder[float32]()
	if 0 < k {
		graphBuilder.SetK(uint(k))
	}
	if 0.0 < rho {
		graphBuilder.SetRho(float64(rho))
	}

	builder := index.NewGraphIndexBuilder[float32](dim, graphBuilder)

	paramString := builder.GetPrameterString()
	if convToBool(useProfile) {
		pprofFileName := getProfileFileName("aknn", "train", paramString)
		callback := startProfiler(pprofFileName)
		defer callback()
	}

	ctx := context.Background()
	idx, err := builder.Build(ctx, features)
	if err != nil {
		panic(err)
	}

	return newSymbol("aknn", idx, paramString, convToBool(useProfile))
}

//export NewRpAKnnIndex
func NewRpAKnnIndex(arr *C.float, arrRows, arrCols C.int, leafs, entriesNum, k C.int, rho C.float, useProfile C._Bool) Symbol {
	features := convertToSlice[float32](arr, arrRows, arrCols)
	dim := uint(len(features[0]))

	rpTreeBuilder := bsp_tree.NewRpTreeBuilder[float32]()
	if 0 < leafs {
		rpTreeBuilder.SetLeafs(uint(leafs))
	}
	rpBuilder := index.NewBspTreeIndexBuilder[float32](dim, rpTreeBuilder)
	rpBuilder.SetTrees(uint(1))

	graphBuilder := graph.NewAKnnGraphBuilder[float32]()
	if 0 < k {
		graphBuilder.SetK(uint(k))
	}
	if 0.0 < rho {
		graphBuilder.SetRho(float64(rho))
	}
	aknnBuilder := index.NewGraphIndexBuilder[float32](dim, graphBuilder)

	builder := index.NewCompositeIndexBuilder[float32, index.BspTreeIndex[float32], index.GraphIndex[float32]](rpBuilder, aknnBuilder)
	if 0 < entriesNum {
		builder.SetEntriesNum(32)
	}

	paramString := builder.GetPrameterString()
	if convToBool(useProfile) {
		pprofFileName := getProfileFileName("rpaknn", "train", paramString)
		callback := startProfiler(pprofFileName)
		defer callback()
	}

	ctx := context.Background()
	idx, err := builder.Build(ctx, features)
	if err != nil {
		panic(err)
	}

	return newSymbol("rpaknn", idx, paramString, convToBool(useProfile))
}

//export Search
func Search(symbol Symbol, arrQuery *C.float, arrRet *C.int, rows, cols, k, n C.int) {
	algo := indexes[symbol]
	if algo.useProfile {
		paramString := fmt.Sprintf("%s_k=%d_n=%d", algo.paramString, k, n)
		pprofFileName := getProfileFileName(algo.name, "predict", paramString)
		callback := startProfiler(pprofFileName)
		defer callback()
	}

	queries := convertToSlice[float32](arrQuery, rows, cols)
	ret := convertToSlice[int32](arrRet, rows, k)

	ctx := context.Background()
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	for i := range queries {
		ch := algo.index.SearchChannel(ctx, queries[i])
		searchResults, err := countrymaam.Search(ch, uint(k), uint(n))
		if err != nil {
			panic(err)
		}

		for j, r := range searchResults {
			ret[i][j] = int32(r.Index)
		}
		for j := len(searchResults); j < int(k); j++ {
			ret[i][j] = -1
		}
	}
}

func main() {}
