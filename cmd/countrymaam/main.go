package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/ar90n/countrymaam"
	"github.com/urfave/cli/v2"
)

type Query struct {
	Feature       []float64
	Neighbors     uint
	MaxCandidates uint
}

func createIndex(index string, nDim uint, leafSize uint, nTrees uint) (countrymaam.Index[float64, int], error) {
	switch index {
	case "flat":
		return countrymaam.NewFlatIndex[float64, int](nDim), nil
	case "kd-tree":
		return countrymaam.NewKdTreeIndex[float64, int](nDim, leafSize), nil
	case "rkd-tree":
		return countrymaam.NewRandomizedKdTreeIndex[float64, int](nDim, leafSize, nTrees), nil
	case "rp-tree":
		return countrymaam.NewRpTreeIndex[float64, int](nDim, leafSize), nil
	case "rrp-tree":
		return countrymaam.NewRandomizedRpTreeIndex[float64, int](nDim, leafSize, nTrees), nil
	default:
		return nil, fmt.Errorf("unknown index name: %s", index)
	}
}

func loadIndex(index string, inputPath string) (countrymaam.Index[float64, int], error) {
	file, err := os.Open(inputPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	switch index {
	case "flat":
		return countrymaam.LoadFlatIndex[float64, int](file)
	case "kd-tree":
		return countrymaam.LoadKdTreeIndex[float64, int](file)
	case "rkd-tree":
		return countrymaam.LoadRandomizedKdTreeIndex[float64, int](file)
	case "rp-tree":
		return countrymaam.LoadRpTreeIndex[float64, int](file)
	case "rrp-tree":
		return countrymaam.LoadRandomizedRpTreeIndex[float64, int](file)
	default:
		return nil, fmt.Errorf("unknown index name: %s", index)
	}
}

func readFeature(r io.Reader, nDim uint) ([]float64, error) {
	feature := make([]float64, nDim)
	for j := uint(0); j < nDim; j++ {
		var v float64
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
	index, err := createIndex(indexName, nDim, leafSize, nTrees)
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
	if err := index.Build(); err != nil {
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
	index, err := loadIndex(indexName, inputName)
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
		neighbors, err := index.Search(query.Feature, query.Neighbors, query.MaxCandidates)
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
