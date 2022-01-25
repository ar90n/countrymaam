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

func createIndex(index string, nDim uint, leafSize uint, nTrees uint, maxCandidates uint) (countrymaam.Index[float64, int], error) {
	switch index {
	case "flat":
		return countrymaam.NewFlatIndex[float64, int](nDim), nil
	case "kd-tree":
		return countrymaam.NewKdTreeIndex[float64, int](nDim, leafSize, maxCandidates), nil
	case "rkd-tree":
		return countrymaam.NewRandomizedKdTreeIndex[float64, int](nDim, leafSize, nTrees, maxCandidates), nil
	case "rp-tree":
		return countrymaam.NewRpTreeIndex[float64, int](nDim, leafSize, maxCandidates), nil
	case "rrp-tree":
		return countrymaam.NewRandomizedRpTreeIndex[float64, int](nDim, leafSize, nTrees, maxCandidates), nil
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

func trainAction(c *cli.Context) error {
	nDim := c.Uint("dim")
	indexName := c.String("index")
	leafSize := c.Uint("leaf-size")
	maxCandidates := c.Uint("max-candidates")
	outputName := c.String("output")
	nTrees := c.Uint("tree-num")
	index, err := createIndex(indexName, nDim, leafSize, nTrees, maxCandidates)
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
	k := c.Uint("k")
	radius := c.Float64("radius")
	indexName := c.String("index")
	inputName := c.String("input")
	index, err := loadIndex(indexName, inputName)
	if err != nil {
		return err
	}

	r := bufio.NewReader(os.Stdin)
Loop:
	for {
		feature, err := readFeature(r, nDim)
		if err == io.EOF {
			break Loop
		}
		if err != nil {
			return err
		}
		neighbors, err := index.Search(feature, k, radius)
		if err != nil {
			return err
		}

		var wtr = bufio.NewWriter(os.Stdout)
		//fmt.Fprintln(os.Stderr, uint32(len(neighbors)))
		binary.Write(wtr, binary.LittleEndian, uint32(len(neighbors)))
		for _, n := range neighbors {
			//fmt.Fprintln(os.Stderr, uint32(n.Item))
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
						Name:  "max-candidates",
						Value: 32,
						Usage: "max candidates",
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
					&cli.Uint64Flag{
						Name:  "k",
						Value: 1,
						Usage: "K",
					},
					&cli.Float64Flag{
						Name:  "radius",
						Value: 1.0,
						Usage: "radius",
					},
				},
			},
		},
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}
