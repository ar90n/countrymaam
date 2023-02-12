package countrymaam_test

import (
	"bytes"
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/cut_plane"
	"github.com/ar90n/countrymaam/index"
	"github.com/ar90n/countrymaam/linalg"
)

func getDataset1() [][]float32 {
	return [][]float32{
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
}

func mustNewTreeIndex[T linalg.Number, U comparable](config index.TreeConfig, cpf index.CutPlaneFactory[T, U]) countrymaam.Index[T, U] {
	index, err := index.NewTreeIndex(config, cpf)
	if err != nil {
		panic(err)
	}

	return index
}

func TestSearchKNNVectors(t *testing.T) {
	type Algorithm struct {
		Name  string
		Index countrymaam.Index[float32, int]
	}

	type TestCase struct {
		Query    [8]float32
		K        uint
		Radius   float64
		Expected []int
	}

	dataset := getDataset1()
	datasetDim := uint(len(dataset[0]))
	for _, alg := range []Algorithm{
		{
			"FlatIndex",
			index.NewFlatIndex[float32, int](datasetDim),
		},
		{
			"KDTreeIndex-Leafs:1-Trees:1",
			mustNewTreeIndex[float32, int](
				index.TreeConfig{
					Dim: datasetDim,
				},
				cut_plane.NewKdCutPlaneFactory[float32, int](0, 0),
			),
		},
		{
			"KDTreeIndex-Leafs:5-Trees:1",
			mustNewTreeIndex[float32, int](
				index.TreeConfig{
					Dim:   datasetDim,
					Leafs: 5,
				},
				cut_plane.NewKdCutPlaneFactory[float32, int](0, 0),
			),
		},
		{
			"KDTreeIndex-Leafs:1-Trees5",
			mustNewTreeIndex[float32, int](
				index.TreeConfig{
					Dim:   datasetDim,
					Trees: 5,
				},
				cut_plane.NewKdCutPlaneFactory[float32, int](100, 5),
			),
		},
		{
			"RpTreeIndex-Leafs:1",
			mustNewTreeIndex[float32, int](
				index.TreeConfig{
					Dim: datasetDim,
				},
				cut_plane.NewRpCutPlaneFactory[float32, int](32),
			),
		},
		{
			"RpTreeIndex-Leafs:5",
			mustNewTreeIndex[float32, int](
				index.TreeConfig{
					Dim:   datasetDim,
					Leafs: 5,
				},
				cut_plane.NewRpCutPlaneFactory[float32, int](32),
			),
		},
		{
			"RpTreeIndex-Leafs:1-Trees:5",
			mustNewTreeIndex[float32, int](
				index.TreeConfig{
					Dim:   datasetDim,
					Trees: 5,
				},
				cut_plane.NewRpCutPlaneFactory[float32, int](32),
			),
		},
	} {
		t.Run(alg.Name, func(t *testing.T) {
			ctx := context.Background()
			for i, data := range dataset {
				data := data
				alg.Index.Add(data[:], i)
			}
			alg.Index.Build(ctx)

			for c, tc := range []TestCase{
				{
					Query:    [8]float32{-0.621, -0.586, -0.468, 0.494, 0.485, 0.407, 1.273, -1.1},
					K:        1,
					Expected: []int{5},
				},
				{
					Query:    [8]float32{-0.83059702, -1.01070708, -0.15162675, -1.32760066, -1.19706362, -0.21952724, -0.27582108, 0.93780233},
					K:        2,
					Expected: []int{0, 9},
				},
				{
					Query:    [8]float32{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					K:        5,
					Expected: []int{2, 4, 5, 7, 9},
				},
				{
					Query:    [8]float32{-0.621, -0.586, -0.468, 0.494, 0.485, 0.407, 1.273, -1.1},
					K:        10,
					Expected: []int{5, 7, 2, 8, 4, 1, 6, 0, 9, 3},
				},
				{
					Query:    [8]float32{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					K:        5,
					Expected: []int{2, 4, 5, 7, 9},
				},
			} {
				t.Run(fmt.Sprint(c), func(t *testing.T) {
					results, _ := alg.Index.Search(ctx, tc.Query[:], tc.K, 64)
					if len(results) != len(tc.Expected) {
						t.Errorf("Expected 1 result, got %d", len(results))
					}

					resultIndice := []int{}
					for _, v := range results {
						resultIndice = append(resultIndice, v.Item)
					}
					if !reflect.DeepEqual(resultIndice, tc.Expected) {
						t.Errorf("Expected results to be %v, got %v", tc.Expected, results)
					}
				})
			}
		})
	}
}

func TestRebuildIndex(t *testing.T) {
	type Algorithm struct {
		Name  string
		Index countrymaam.Index[float32, int]
	}

	type TestCase struct {
		Query    [8]float32
		K        uint
		Radius   float64
		Expected []int
	}

	dataset := getDataset1()
	datasetDim := uint(len(dataset[0]))
	for _, alg := range []Algorithm{
		{
			"FlatIndex",
			index.NewFlatIndex[float32, int](datasetDim),
		},
		{
			"KDTreeIndex",
			mustNewTreeIndex[float32, int](index.TreeConfig{Dim: datasetDim}, cut_plane.NewKdCutPlaneFactory[float32, int](0, 0)),
		},
		{
			"RandomizedKDTreeIndex",
			mustNewTreeIndex[float32, int](
				index.TreeConfig{
					Dim:   datasetDim,
					Leafs: 1,
					Trees: 5,
				},
				cut_plane.NewKdCutPlaneFactory[float32, int](100, 5),
			),
		},
	} {
		t.Run(alg.Name, func(t *testing.T) {
			ctx := context.Background()
			nData := len(dataset)
			nInitialData := nData / 2
			for i := 0; i < nInitialData; i++ {
				alg.Index.Add(dataset[i], i)
			}
			alg.Index.Build(ctx)

			if !alg.Index.HasIndex() {
				t.Error("Index should have been built")
			}

			for i := nInitialData; i < nData; i++ {
				alg.Index.Add(dataset[i], i)
			}

			for c, tc := range []TestCase{
				{
					Query:    [8]float32{-0.621, -0.586, -0.468, 0.494, 0.485, 0.407, 1.273, -1.1},
					K:        1,
					Expected: []int{5},
				},
				{
					Query:    [8]float32{-0.83059702, -1.01070708, -0.15162675, -1.32760066, -1.19706362, -0.21952724, -0.27582108, 0.93780233},
					K:        2,
					Expected: []int{0, 9},
				},
				{
					Query:    [8]float32{-0.621, -0.586, -0.468, 0.494, 0.485, 0.407, 1.273, -1.1},
					K:        10,
					Expected: []int{5, 7, 2, 8, 4, 1, 6, 0, 9, 3},
				},
			} {
				t.Run(fmt.Sprint(c), func(t *testing.T) {
					results, _ := alg.Index.Search(ctx, tc.Query[:], tc.K, 64)
					if len(results) != len(tc.Expected) {
						t.Errorf("Expected 1 result, got %d", len(results))
					}

					resultIndice := []int{}
					for _, v := range results {
						resultIndice = append(resultIndice, v.Item)
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
		Index    countrymaam.Index[float32, int]
		Expected bool
	}

	dataset := getDataset1()
	datasetDim := uint(len(dataset[0]))
	for _, alg := range []TestCase{
		{
			"FlatIndex",
			index.NewFlatIndex[float32, int](datasetDim),
			true,
		},
		{
			"KDTreeIndex",
			mustNewTreeIndex[float32, int](
				index.TreeConfig{
					Dim:   datasetDim,
					Leafs: 1,
				},
				cut_plane.NewKdCutPlaneFactory[float32, int](0, 0),
			),
			false,
		},
		{
			"RandomizedKDTreeIndex",
			mustNewTreeIndex[float32, int](
				index.TreeConfig{
					Dim:   datasetDim,
					Leafs: 1,
					Trees: 5,
				},
				cut_plane.NewKdCutPlaneFactory[float32, int](0, 0),
			),
			false,
		},
	} {
		t.Run(alg.Name, func(t *testing.T) {
			ctx := context.Background()
			err := alg.Index.Build(ctx)
			if (err == nil) != alg.Expected {
				t.Errorf("Expected error to be %v, got %v", alg.Expected, err)
			}
		})
	}
}

func testSerDes[T linalg.Number, I countrymaam.Index[T, int]](t *testing.T, ind I, dataset [][]T) error {
	ctx := context.Background()
	for i, data := range dataset {
		data := data
		ind.Add(data[:], i)
	}
	ind.Build(ctx)

	buf := make([]byte, 0)
	byteBuffer := bytes.NewBuffer(buf)
	if err := ind.Save(byteBuffer); err != nil {
		return err
	}

	ind2, err := index.LoadFlatIndex[float32, int](byteBuffer)
	if err != nil {
		return err
	}

	if !reflect.DeepEqual(ind, ind2) {
		return fmt.Errorf("Expected %v, got %v", ind, ind2)
	}

	return nil
}

func TestSerDesKNNVectors(t *testing.T) {
	dataset := [][]float32{
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
	datasetDim := uint(len(dataset[0]))

	t.Run("FlatIndex", func(t *testing.T) {
		testSerDes(t, index.NewFlatIndex[float32, int](datasetDim), dataset)
	})
	t.Run("KdTreeIndex-Leafs:1", func(t *testing.T) {
		testSerDes(t, mustNewTreeIndex[float32, int](index.TreeConfig{Dim: datasetDim, Leafs: 1}, cut_plane.NewKdCutPlaneFactory[float32, int](0, 0)), dataset)
	})
	t.Run("KDTreeIndex-Leafs:5", func(t *testing.T) {
		testSerDes(t, mustNewTreeIndex[float32, int](index.TreeConfig{Dim: datasetDim, Leafs: 5}, cut_plane.NewKdCutPlaneFactory[float32, int](0, 0)), dataset)
	})
	t.Run("KDTreeIndex-Leafs:1-Trees:5", func(t *testing.T) {
		testSerDes(t, mustNewTreeIndex[float32, int](index.TreeConfig{Dim: datasetDim, Leafs: 1, Trees: 5}, cut_plane.NewKdCutPlaneFactory[float32, int](100, 5)), dataset)
	})
	t.Run("RpTreeIndex-Leafs:1", func(t *testing.T) {
		testSerDes(t, mustNewTreeIndex[float32, int](index.TreeConfig{Dim: datasetDim, Leafs: 1}, cut_plane.NewRpCutPlaneFactory[float32, int](0)), dataset)
	})
	t.Run("RpTreeIndex-Leafs:5", func(t *testing.T) {
		testSerDes(t, mustNewTreeIndex[float32, int](index.TreeConfig{Dim: datasetDim, Leafs: 5}, cut_plane.NewRpCutPlaneFactory[float32, int](0)), dataset)
	})
	t.Run("RpTreeIndex-Leafs:1-Trees:5", func(t *testing.T) {
		testSerDes(t, mustNewTreeIndex[float32, int](index.TreeConfig{Dim: datasetDim, Leafs: 1, Trees: 5}, cut_plane.NewRpCutPlaneFactory[float32, int](32)), dataset)
	})
}
