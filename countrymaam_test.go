package countrymaam_test

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"reflect"
	"testing"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/bsp_tree"
	"github.com/ar90n/countrymaam/graph"
	"github.com/ar90n/countrymaam/index"
	"github.com/stretchr/testify/assert"
)

func getDataset1() [][]float32 {
	features := [][]float32{
		{-0.662, -0.405, 0.508, -0.991, -0.614, -1.639, 0.637, 0.715},
		{0.44, -1.795, -0.243, -1.375, 1.154, 0.142, -0.219, -0.711},
		{0.22, -0.029, 0.7, -0.963, 0.257, 0.419, 0.491, -0.87},
		{0.906, 0.551, -1.198, 1.517, 1.616, 0.014, -1.358, -1.004},
		{0.687, 0.818, 0.868, 0.688, 0.428, 0.582, -0.352, -0.269},
		{-0.621, -0.586, -0.468, 0.494, 0.485, 0.407, 1.273, -1.1},
		{1.606, 1.256, -0.644, -0.858, 0.743, -0.063, 0.042, -1.539},
		{0.255, 1.018, -0.835, -0.288, 0.992, -0.17, 0.764, -1.0},
		{1.061, -0.506, -1.467, 0.043, 1.121, 1.03, 0.596, -1.747},
		{-0.269, -0.346, -0.076, -0.392, 0.301, -1.097, 0.139, 1.692},
		{-1.034, -1.709, -2.693, 1.539, -1.186, 0.29, -0.935, -0.546},
		{1.954, -1.708, -0.423, -2.241, 1.272, -0.253, -1.013, -0.382},
	}
	return features
}

func TestSearchKNNVectors2(t *testing.T) {
	type Algorithm struct {
		Name  string
		Build func(ctx context.Context, features [][]float32) countrymaam.Index[float32]
	}

	type TestCase struct {
		Query    [8]float32
		K        uint
		Radius   float64
		Expected []uint
	}

	dataset := getDataset1()
	datasetDim := uint(len(dataset[0]))
	for _, alg := range []Algorithm{
		{
			"FlatIndex",
			func(ctx context.Context, features [][]float32) countrymaam.Index[float32] {
				builder := index.NewFlatIndexBuilder[float32](datasetDim)
				index, err := builder.Build(context.Background(), features)
				if err != nil {
					panic(err)
				}
				return index
			},
		},
		{
			"KDTreeIndex-Leafs:1-Trees:1",
			func(ctx context.Context, features [][]float32) countrymaam.Index[float32] {
				kdTreeBuilder := bsp_tree.NewKdTreeBuilder[float32]()
				kdTreeBuilder.SetLeafs(1)
				builder := index.NewBspTreeIndexBuilder[float32](datasetDim, kdTreeBuilder)
				builder.SetTrees(1)
				index, err := builder.Build(context.Background(), features)
				if err != nil {
					panic(err)
				}
				return index
			},
		},
		{
			"KDTreeIndex-Leafs:5-Trees:1",
			func(ctx context.Context, features [][]float32) countrymaam.Index[float32] {
				kdTreeBuilder := bsp_tree.NewKdTreeBuilder[float32]()
				kdTreeBuilder.SetLeafs(5)
				builder := index.NewBspTreeIndexBuilder[float32](datasetDim, kdTreeBuilder)
				builder.SetTrees(1)
				index, err := builder.Build(context.Background(), features)
				if err != nil {
					panic(err)
				}
				return index
			},
		},
		{
			"KDTreeIndex-Leafs:1-Trees5",
			func(ctx context.Context, features [][]float32) countrymaam.Index[float32] {
				kdTreeBuilder := bsp_tree.NewKdTreeBuilder[float32]()
				kdTreeBuilder.SetLeafs(1)
				builder := index.NewBspTreeIndexBuilder[float32](datasetDim, kdTreeBuilder)
				builder.SetTrees(5)
				index, err := builder.Build(context.Background(), features)
				if err != nil {
					panic(err)
				}
				return index
			},
		},
		{
			"RpTreeIndex-Leafs:1",
			func(ctx context.Context, features [][]float32) countrymaam.Index[float32] {
				rpTreeBuilder := bsp_tree.NewRpTreeBuilder[float32]()
				rpTreeBuilder.SetLeafs(1)
				builder := index.NewBspTreeIndexBuilder[float32](datasetDim, rpTreeBuilder)
				index, err := builder.Build(context.Background(), features)
				if err != nil {
					panic(err)
				}
				return index
			},
		},
		{
			"RpTreeIndex-Leafs:5",
			func(ctx context.Context, features [][]float32) countrymaam.Index[float32] {
				rpTreeBuilder := bsp_tree.NewRpTreeBuilder[float32]()
				rpTreeBuilder.SetLeafs(5)
				builder := index.NewBspTreeIndexBuilder[float32](datasetDim, rpTreeBuilder)
				index, err := builder.Build(context.Background(), features)
				if err != nil {
					panic(err)
				}
				return index
			},
		},
		{
			"RpTreeIndex-Leafs:1-Trees:5",
			func(ctx context.Context, features [][]float32) countrymaam.Index[float32] {
				rpTreeBuilder := bsp_tree.NewRpTreeBuilder[float32]()
				rpTreeBuilder.SetLeafs(1)
				builder := index.NewBspTreeIndexBuilder[float32](datasetDim, rpTreeBuilder)
				builder.SetTrees(5)
				index, err := builder.Build(context.Background(), features)
				if err != nil {
					panic(err)
				}
				return index
			},
		},
		{
			"AKnnGraphIndex",
			func(ctx context.Context, features [][]float32) countrymaam.Index[float32] {
				graphBuilder := graph.NewAKnnGraphBuilder[float32]()
				graphBuilder.SetK(3).SetRho(0.3)
				builder := index.NewGraphIndexBuilder[float32](datasetDim, graphBuilder)
				index, err := builder.Build(context.Background(), features)
				if err != nil {
					panic(err)
				}
				return index
			},
		},
		{
			"RpAKnnGraphIndex",
			func(ctx context.Context, features [][]float32) countrymaam.Index[float32] {
				rpTreeBuilder := bsp_tree.NewRpTreeBuilder[float32]()
				rpBuilder := index.NewBspTreeIndexBuilder[float32](datasetDim, rpTreeBuilder)
				rpBuilder.SetTrees(1)

				graphBuilder := graph.NewAKnnGraphBuilder[float32]()
				graphBuilder.SetK(3).SetRho(0.3)
				aknnBuilder := index.NewGraphIndexBuilder[float32](datasetDim, graphBuilder)

				builder := index.NewCompositeIndexBuilder[float32, index.BspTreeIndex[float32], index.GraphIndex[float32]](rpBuilder, aknnBuilder)
				index, err := builder.Build(context.Background(), features)
				if err != nil {
					panic(err)
				}
				return index
			},
		},
	} {
		t.Run(alg.Name, func(t *testing.T) {
			features := getDataset1()
			ctx := context.Background()
			index := alg.Build(ctx, features)

			for c, tc := range []TestCase{
				{
					Query:    [8]float32{-0.621, -0.586, -0.468, 0.494, 0.485, 0.407, 1.273, -1.1},
					K:        1,
					Expected: []uint{5},
				},
				{
					Query:    [8]float32{-0.83059702, -1.01070708, -0.15162675, -1.32760066, -1.19706362, -0.21952724, -0.27582108, 0.93780233},
					K:        2,
					Expected: []uint{0, 9},
				},
				{
					Query:    [8]float32{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					K:        5,
					Expected: []uint{2, 4, 5, 7, 9},
				},
				{
					Query:    [8]float32{-0.621, -0.586, -0.468, 0.494, 0.485, 0.407, 1.273, -1.1},
					K:        10,
					Expected: []uint{5, 7, 2, 8, 4, 1, 6, 0, 9, 3},
				},
				{
					Query:    [8]float32{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					K:        5,
					Expected: []uint{2, 4, 5, 7, 9},
				},
			} {
				t.Run(fmt.Sprint(c), func(t *testing.T) {
					ch := index.SearchChannel(ctx, tc.Query[:])
					results, _ := countrymaam.Search(ch, tc.K, 64)
					if len(results) != len(tc.Expected) {
						t.Errorf("Expected 1 result, got %d", len(results))
					}

					resultIndice := []uint{}
					for _, v := range results {
						resultIndice = append(resultIndice, v.Index)
					}
					if !reflect.DeepEqual(resultIndice, tc.Expected) {
						t.Errorf("Expected results to be %v, got %v", tc.Expected, results)
					}
				})
			}
		})
	}
}

func TestBuildIndexWhenPoolIsEmpty(t *testing.T) {
	type TestCase struct {
		Name     string
		Build    func(ctx context.Context, features [][]float32, items []int) error
		Expected bool
	}

	datasetDim := uint(8)
	for _, alg := range []TestCase{
		{
			"FlatIndex",
			func(ctx context.Context, features [][]float32, items []int) error {
				builder := index.NewFlatIndexBuilder[float32](datasetDim)
				_, err := builder.Build(context.Background(), features)
				return err
			},
			true,
		},
		{
			"BspTreeIndex",
			func(ctx context.Context, features [][]float32, items []int) error {
				kdTreeBuilder := bsp_tree.NewKdTreeBuilder[float32]()
				builder := index.NewBspTreeIndexBuilder[float32](datasetDim, kdTreeBuilder)
				_, err := builder.Build(context.Background(), features)
				return err
			},
			true,
		},
		{
			"GraphIndex",
			func(ctx context.Context, features [][]float32, items []int) error {
				graphBuilder := graph.NewAKnnGraphBuilder[float32]()
				builder := index.NewGraphIndexBuilder[float32](datasetDim, graphBuilder)
				_, err := builder.Build(context.Background(), features)
				return err
			},
			true,
		},
	} {
		t.Run(alg.Name, func(t *testing.T) {
			ctx := context.Background()
			err := alg.Build(ctx, [][]float32{}, []int{})
			if (err == nil) != alg.Expected {
				t.Errorf("Expected error to be %v, got %v", alg.Expected, err)
			}
		})
	}
}

func testSerDes[I countrymaam.Index[float32]](t *testing.T, ind *I, loadFunc func(r io.Reader) (*I, error)) error {
	buf := make([]byte, 0)
	byteBuffer := bytes.NewBuffer(buf)
	if err := (*ind).Save(byteBuffer); err != nil {
		return err
	}

	ind2, err := loadFunc(byteBuffer)
	if err != nil {
		return err
	}

	if !reflect.DeepEqual(*ind, *ind2) {
		return fmt.Errorf("Expected %v, got %v", ind, ind2)
	}

	ch := (*ind).SearchChannel(context.Background(), []float32{1.606, 1.256, -0.644, -0.858, 0.743, -0.063, 0.042, -1.539})
	a := <-ch
	assert.Equal(t, uint(6), a.Index)

	return nil
}

func TestSerDesKNNVectors(t *testing.T) {
	features := [][]float32{
		{-0.662, -0.405, 0.508, -0.991, -0.614, -1.639, 0.637, 0.715},
		{0.44, -1.795, -0.243, -1.375, 1.154, 0.142, -0.219, -0.711},
		{0.22, -0.029, 0.7, -0.963, 0.257, 0.419, 0.491, -0.87},
		{0.906, 0.551, -1.198, 1.517, 1.616, 0.014, -1.358, -1.004},
		{0.687, 0.818, 0.868, 0.688, 0.428, 0.582, -0.352, -0.269},
		{-0.621, -0.586, -0.468, 0.494, 0.485, 0.407, 1.273, -1.1},
		{1.606, 1.256, -0.644, -0.858, 0.743, -0.063, 0.042, -1.539},
		{0.255, 1.018, -0.835, -0.288, 0.992, -0.17, 0.764, -1.0},
		{1.061, -0.506, -1.467, 0.043, 1.121, 1.03, 0.596, -1.747},
		{-0.269, -0.346, -0.076, -0.392, 0.301, -1.097, 0.139, 1.692},
		{-1.034, -1.709, -2.693, 1.539, -1.186, 0.29, -0.935, -0.546},
		{1.954, -1.708, -0.423, -2.241, 1.272, -0.253, -1.013, -0.382},
	}
	datasetDim := uint(len(features[0]))

	t.Run("FlatIndex", func(t *testing.T) {
		builder := index.NewFlatIndexBuilder[float32](datasetDim)
		ind, _ := builder.Build(context.Background(), features)
		loadFunc := func(r io.Reader) (*index.FlatIndex[float32], error) {
			return index.LoadFlatIndex[float32](r)
		}
		testSerDes(t, ind, loadFunc)
	})
	t.Run("KdTreeIndex-Leafs:1", func(t *testing.T) {
		kdTreeBuilder := bsp_tree.NewKdTreeBuilder[float32]()
		kdTreeBuilder.SetSampleFeatures(100).SetTopKCandidates(5).SetLeafs(1)

		builder := index.NewBspTreeIndexBuilder[float32](datasetDim, kdTreeBuilder)
		ind, _ := builder.Build(context.Background(), features)
		loadFunc := func(r io.Reader) (*index.BspTreeIndex[float32], error) {
			return index.LoadBspTreeIndex[float32](r)
		}
		testSerDes(t, ind, loadFunc)
	})
	t.Run("RpTreeIndex-Leafs:1", func(t *testing.T) {
		rpTreeBuilder := bsp_tree.NewRpTreeBuilder[float32]()
		rpTreeBuilder.SetSampleFeatures(32).SetLeafs(1)

		builder := index.NewBspTreeIndexBuilder[float32](datasetDim, rpTreeBuilder)
		ind, _ := builder.Build(context.Background(), features)
		loadFunc := func(r io.Reader) (*index.BspTreeIndex[float32], error) {
			return index.LoadBspTreeIndex[float32](r)
		}
		testSerDes(t, ind, loadFunc)
	})

	t.Run("GraphIndex", func(t *testing.T) {
		graphBuilder := graph.NewAKnnGraphBuilder[float32]()
		graphBuilder.SetK(2)
		builder := index.NewGraphIndexBuilder[float32](datasetDim, graphBuilder)
		ind, _ := builder.Build(context.Background(), features)
		loadFunc := func(r io.Reader) (*index.GraphIndex[float32], error) {
			return index.LoadGraphIndex[float32](r)
		}
		testSerDes(t, ind, loadFunc)
	})

	t.Run("ComposeIndex", func(t *testing.T) {
		rpTreeBuilder := bsp_tree.NewRpTreeBuilder[float32]()
		rpTreeBuilder.SetSampleFeatures(32).SetLeafs(8)
		rpBuilder := index.NewBspTreeIndexBuilder[float32](datasetDim, rpTreeBuilder)
		rpBuilder.SetTrees(1)

		graphBuilder := graph.NewAKnnGraphBuilder[float32]()
		graphBuilder.SetK(4).SetRho(0.3)
		aknnBuilder := index.NewGraphIndexBuilder[float32](datasetDim, graphBuilder)

		builder := index.NewCompositeIndexBuilder[float32, index.BspTreeIndex[float32], index.GraphIndex[float32]](rpBuilder, aknnBuilder)
		ind, _ := builder.Build(context.Background(), features)

		loadFunc := func(r io.Reader) (*index.CompositeIndex[float32], error) {
			return index.LoadCompositeIndex[float32](r)
		}
		testSerDes(t, ind, loadFunc)
	})
}
