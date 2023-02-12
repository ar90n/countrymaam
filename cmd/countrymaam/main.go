package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
	"runtime/pprof"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/cut_plane"
	"github.com/ar90n/countrymaam/index"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/urfave/cli/v2"
)

type Query[T linalg.Number] struct {
	Feature       []T
	Neighbors     uint
	MaxCandidates uint
}

func createIndex[T linalg.Number, U comparable](ind string, nDim uint, leafSize uint, nTrees uint) (countrymaam.Index[T, U], error) {
	switch ind {
	case "flat":
		return index.NewFlatIndex[T, U](nDim), nil
	case "kd-tree":
		return index.NewTreeIndex(
			index.TreeConfig{
				Dim:   nDim,
				Leafs: leafSize,
			},
			cut_plane.NewKdCutPlaneFactory[T, U](0, 0),
		)
	case "rkd-tree":
		return index.NewTreeIndex(
			index.TreeConfig{
				Dim:   nDim,
				Leafs: leafSize,
				Trees: nTrees,
			},
			cut_plane.NewKdCutPlaneFactory[T, U](100, 5),
		)
	case "rp-tree":
		return index.NewTreeIndex(
			index.TreeConfig{
				Dim:   nDim,
				Leafs: leafSize,
			},
			cut_plane.NewRpCutPlaneFactory[T, U](0),
		)
	case "rrp-tree":
		return index.NewTreeIndex(
			index.TreeConfig{
				Dim:   nDim,
				Leafs: leafSize,
				Trees: nTrees,
			},
			cut_plane.NewRpCutPlaneFactory[T, U](32),
		)
	default:
		return nil, fmt.Errorf("unknown index name: %s", ind)
	}
}

func loadIndex[T linalg.Number, U comparable](ind string, inputPath string) (countrymaam.Index[T, U], error) {
	file, err := os.Open(inputPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	switch ind {
	case "flat":
		return index.LoadFlatIndex[T, U](file)
	case "kd-tree":
		return index.LoadTreeIndex(file, cut_plane.NewKdCutPlaneFactory[T, U](0, 0))
	case "rkd-tree":
		return index.LoadTreeIndex(file, cut_plane.NewKdCutPlaneFactory[T, U](0, 0))
	case "rp-tree":
		return index.LoadTreeIndex(file, cut_plane.NewRpCutPlaneFactory[T, U](0))
	case "rrp-tree":
		return index.LoadTreeIndex(file, cut_plane.NewRpCutPlaneFactory[T, U](0))
	default:
		return nil, fmt.Errorf("unknown index name: %s", ind)
	}
}

func readFeature[T linalg.Number](r io.Reader, nDim uint) ([]T, error) {
	feature := make([]T, nDim)
	for j := uint(0); j < nDim; j++ {
		var v T
		err := binary.Read(r, binary.LittleEndian, &v)
		if err != nil {
			return nil, err
		}
		feature[j] = v
	}

	return feature, nil
}

func readQuery[T linalg.Number](r io.Reader, nDim uint) (Query[T], error) {
	var maxCandidates int32
	if err := binary.Read(r, binary.LittleEndian, &maxCandidates); err != nil {
		return Query[T]{}, err
	}

	var neighbors int32
	if err := binary.Read(r, binary.LittleEndian, &neighbors); err != nil {
		return Query[T]{}, err
	}

	feature, err := readFeature[T](r, nDim)
	if err != nil {
		return Query[T]{}, err
	}

	query := Query[T]{
		Feature:       feature,
		Neighbors:     uint(neighbors),
		MaxCandidates: uint(maxCandidates),
	}
	return query, nil
}

func trainAction(c *cli.Context) error {
	dtype := c.String("dtype")
	nDim := c.Uint("dim")
	indexName := c.String("index")
	leafSize := c.Uint("leaf-size")
	outputName := c.String("output")
	nTrees := c.Uint("tree-num")
	profileOutputName := c.String("profile-output")

	switch dtype {
	case "float32":
		return train[float32](nDim, indexName, leafSize, outputName, nTrees, profileOutputName)
	case "uint8":
		return train[uint8](nDim, indexName, leafSize, outputName, nTrees, profileOutputName)
	default:
		return fmt.Errorf("unknown dtype: %s", dtype)
	}
}

func train[T linalg.Number](nDim uint, indexName string, leafSize uint, outputName string, nTrees uint, profileOutputName string) error {
	if profileOutputName != "" {
		f, err := os.Create(profileOutputName)
		if err != nil {
			return err
		}
		defer f.Close()

		if err := pprof.StartCPUProfile(f); err != nil {
			return err
		}
		defer pprof.StopCPUProfile()
	}

	ctx := context.Background()
	index, err := createIndex[T, int](indexName, nDim, leafSize, nTrees)
	if err != nil {
		return err
	}

	log.Println("reading data...")
	r := bufio.NewReader(os.Stdin)
Loop:
	for i := 0; ; i++ {
		feature, err := readFeature[T](r, nDim)
		if err == io.EOF {
			break Loop
		}
		if err != nil {
			return err
		}
		index.Add(feature, i)
	}
	log.Println("done")

	log.Println("building index...")
	if err := index.Build(ctx); err != nil {
		return err
	}
	log.Println("done")

	log.Println("saving index...")
	file, err := os.Create(outputName)
	if err != nil {
		return err
	}
	defer file.Close()
	if err := index.Save(file); err != nil {
		return err
	}
	log.Println("done")

	return nil
}

func predictAction(c *cli.Context) error {
	dtype := c.String("dtype")
	nDim := c.Uint("dim")
	indexName := c.String("index")
	inputName := c.String("input")
	profileOutputName := c.String("profile-output")

	switch dtype {
	case "float32":
		return predict[float32](nDim, indexName, inputName, profileOutputName)
	case "uint8":
		return predict[uint8](nDim, indexName, inputName, profileOutputName)
	default:
		return fmt.Errorf("unknown dtype: %s", dtype)
	}

}

func predict[T linalg.Number](nDim uint, indexName string, inputName string, profileOutputName string) error {
	if profileOutputName != "" {
		f, err := os.Create(profileOutputName)
		if err != nil {
			return err
		}
		defer f.Close()

		if err := pprof.StartCPUProfile(f); err != nil {
			return err
		}
		defer pprof.StopCPUProfile()
	}

	ctx := context.Background()
	index, err := loadIndex[T, int](indexName, inputName)
	if err != nil {
		return err
	}

	r := bufio.NewReader(os.Stdin)
Loop:
	for {
		query, err := readQuery[T](r, nDim)
		if err == io.EOF {
			break Loop
		}
		if err != nil {
			return err
		}
		neighbors, err := index.Search(ctx, query.Feature, query.Neighbors, query.MaxCandidates)
		if err != nil {
			return err
		}

		var wtr = bufio.NewWriter(os.Stdout)
		binary.Write(wtr, binary.LittleEndian, uint32(len(neighbors)))
		for _, n := range neighbors {
			binary.Write(wtr, binary.LittleEndian, uint32(n.Item))
		}
		wtr.Flush()
	}

	return nil
}

func main() {
	app := &cli.App{
		Name:     "countrymaam",
		HelpName: "countrymaam",
		Usage:    "benchmark program for countrymaam",
		Commands: []*cli.Command{
			{
				Name:      "train",
				Usage:     "train index",
				UsageText: "countrymaam train [command options]",
				Action:    trainAction,
				Flags: []cli.Flag{
					&cli.UintFlag{
						Name:  "dim",
						Value: 32,
						Usage: "dimension of feature",
					},
					&cli.StringFlag{
						Name:  "dtype",
						Value: "float32",
						Usage: "data type",
					},
					&cli.StringFlag{
						Name:  "index",
						Value: "flat",
						Usage: "index type",
					},
					&cli.UintFlag{
						Name:  "leaf-size",
						Value: 1,
						Usage: "leaf size",
					},
					&cli.UintFlag{
						Name:  "tree-num",
						Value: 8,
						Usage: "number of trees",
					},
					&cli.StringFlag{
						Name:  "output",
						Value: "index.bin",
						Usage: "output file",
					},
					&cli.StringFlag{
						Name:  "profile-output",
						Value: "cpu.pprof",
						Usage: "profile output file",
					},
				},
			},
			{
				Name:      "predict",
				Usage:     "predict neighbors",
				UsageText: "countrymaam predict [command options]",
				Action:    predictAction,
				Flags: []cli.Flag{
					&cli.UintFlag{
						Name:  "dim",
						Value: 32,
						Usage: "dimension of feature",
					},
					&cli.StringFlag{
						Name:  "dtype",
						Value: "float32",
						Usage: "data type",
					},
					&cli.StringFlag{
						Name:  "index",
						Value: "flat",
						Usage: "index type",
					},
					&cli.StringFlag{
						Name:  "input",
						Value: "index.bin",
						Usage: "index file",
					},
					&cli.StringFlag{
						Name:  "profile-output",
						Value: "cpu.pprof",
						Usage: "profile output file",
					},
				},
			},
		},
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}
