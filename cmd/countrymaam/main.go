package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/urfave/cli/v2"
)

type Query struct {
	Feature       []float32
	Neighbors     uint
	MaxCandidates uint
}

func createIndex(index string, nDim uint, leafSize uint, nTrees uint, opts linalg.LinAlgOptions) (countrymaam.Index[float32, int], error) {
	switch index {
	case "flat":
		return countrymaam.NewFlatIndex[float32, int](nDim, opts), nil
	case "kd-tree":
		return countrymaam.NewKdTreeIndex[float32, int](nDim, leafSize, opts), nil
	case "rkd-tree":
		return countrymaam.NewRandomizedKdTreeIndex[float32, int](nDim, leafSize, nTrees, opts), nil
	case "rp-tree":
		return countrymaam.NewRpTreeIndex[float32, int](nDim, leafSize, opts), nil
	case "rrp-tree":
		return countrymaam.NewRandomizedRpTreeIndex[float32, int](nDim, leafSize, nTrees, opts), nil
	default:
		return nil, fmt.Errorf("unknown index name: %s", index)
	}
}

func loadIndex(index string, inputPath string, opts linalg.LinAlgOptions) (countrymaam.Index[float32, int], error) {
	file, err := os.Open(inputPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	switch index {
	case "flat":
		return countrymaam.LoadFlatIndex[float32, int](file, opts)
	case "kd-tree":
		return countrymaam.LoadKdTreeIndex[float32, int](file, opts)
	case "rkd-tree":
		return countrymaam.LoadRandomizedKdTreeIndex[float32, int](file, opts)
	case "rp-tree":
		return countrymaam.LoadRpTreeIndex[float32, int](file, opts)
	case "rrp-tree":
		return countrymaam.LoadRandomizedRpTreeIndex[float32, int](file, opts)
	default:
		return nil, fmt.Errorf("unknown index name: %s", index)
	}
}

func readFeature(r io.Reader, nDim uint) ([]float32, error) {
	feature := make([]float32, nDim)
	for j := uint(0); j < nDim; j++ {
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		if err != nil {
			return nil, err
		}
		feature[j] = v
	}

	return feature, nil
}

func readQuery(r io.Reader, nDim uint) (Query, error) {
	var maxCandidates int32
	if err := binary.Read(r, binary.LittleEndian, &maxCandidates); err != nil {
		return Query{}, err
	}

	var neighbors int32
	if err := binary.Read(r, binary.LittleEndian, &neighbors); err != nil {
		return Query{}, err
	}

	feature, err := readFeature(r, nDim)
	if err != nil {
		return Query{}, err
	}

	query := Query{
		Feature:       feature,
		Neighbors:     uint(neighbors),
		MaxCandidates: uint(maxCandidates),
	}
	return query, nil
}

func trainAction(c *cli.Context) error {
	nDim := c.Uint("dim")
	indexName := c.String("index")
	leafSize := c.Uint("leaf-size")
	outputName := c.String("output")
	nTrees := c.Uint("tree-num")

	ctx := context.Background()
	opts := linalg.LinAlgOptions{UseAVX2: true}
	index, err := createIndex(indexName, nDim, leafSize, nTrees, opts)
	if err != nil {
		return err
	}

	log.Println("reading data...")
	r := bufio.NewReader(os.Stdin)
Loop:
	for i := 0; ; i++ {
		feature, err := readFeature(r, nDim)
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
	nDim := c.Uint("dim")
	indexName := c.String("index")
	inputName := c.String("input")

	ctx := context.Background()
	opts := linalg.LinAlgOptions{UseAVX2: true}
	index, err := loadIndex(indexName, inputName, opts)
	if err != nil {
		return err
	}

	r := bufio.NewReader(os.Stdin)
Loop:
	for {
		query, err := readQuery(r, nDim)
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
						Name:  "index",
						Value: "flat",
						Usage: "dimension of feature",
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
						Name:  "index",
						Value: "flat",
						Usage: "dimension of feature",
					},
					&cli.StringFlag{
						Name:  "input",
						Value: "index.bin",
						Usage: "index file",
					},
				},
			},
		},
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}
