package countrymaam

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/ar90n/countrymaam/index"
	"github.com/ar90n/countrymaam/metric"
)

func TestSearchKNNVectors(t *testing.T) {
	dataset := [...][8]float32{
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

	type Algorithm struct {
		Name  string
		Index Index[float32, int]
	}

	type TestCase struct {
		Query    [8]float32
		K        uint
		Radius   float32
		Expected []int
	}

	datasetDim := uint(len(dataset[0]))
	for _, alg := range []Algorithm{
		{
			"FlatIndex",
			index.NewFlatIndex[float32, int, metric.SqL2Dist[float32]](datasetDim),
		},
		{
			"KDTreeIndex-lefSize:1",
			index.NewKdTreeIndex[float32, int, metric.SqL2Dist[float32]](datasetDim, 1),
		},
		{
			"KDTreeIndex-leafSize:5",
			index.NewKdTreeIndex[float32, int, metric.SqL2Dist[float32]](datasetDim, 5),
		},
		{
			"RandomizedKDTreeIndex-lefSize:1-5",
			index.NewRandomizedKdTreeIndex[float32, int, metric.SqL2Dist[float32]](datasetDim, 1, 5),
		},
		{
			"RpTreeIndex-lefSize:1",
			index.NewRpTreeIndex[float32, int, metric.SqL2Dist[float32]](datasetDim, 1),
		},
		{
			"RpTreeIndex-lefSize:5",
			index.NewRpTreeIndex[float32, int, metric.SqL2Dist[float32]](datasetDim, 5),
		},
		{
			"RandomizedRpTreeIndex-lefSize:1-5",
			index.NewRandomizedRpTreeIndex[float32, int, metric.SqL2Dist[float32]](datasetDim, 1, 5),
		},
	} {
		t.Run(alg.Name, func(t *testing.T) {
			for i, data := range dataset {
				data := data
				alg.Index.Add(data[:], i)
			}
			alg.Index.Build()

			for c, tc := range []TestCase{
				{
					Query:    [8]float32{-0.621, -0.586, -0.468, 0.494, 0.485, 0.407, 1.273, -1.1},
					K:        1,
					Radius:   0.5,
					Expected: []int{5},
				},
				{
					Query:    [8]float32{-0.83059702, -1.01070708, -0.15162675, -1.32760066, -1.19706362, -0.21952724, -0.27582108, 0.93780233},
					K:        2,
					Radius:   10,
					Expected: []int{0, 9},
				},
				{
					Query:    [8]float32{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					K:        5,
					Radius:   15,
					Expected: []int{2, 4, 5, 7, 9},
				},
				{
					Query:    [8]float32{-0.621, -0.586, -0.468, 0.494, 0.485, 0.407, 1.273, -1.1},
					K:        10,
					Radius:   1e-8,
					Expected: []int{5},
				},
				{
					Query:    [8]float32{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
					K:        5,
					Radius:   0.5,
					Expected: []int{},
				},
			} {
				t.Run(fmt.Sprint(c), func(t *testing.T) {
					results, _ := alg.Index.Search(tc.Query[:], tc.K, tc.Radius)
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
