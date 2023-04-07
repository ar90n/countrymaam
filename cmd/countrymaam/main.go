package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"runtime/pprof"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/bsp_tree"
	"github.com/ar90n/countrymaam/graph"
	"github.com/ar90n/countrymaam/index"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/urfave/cli/v2"
)

type Query[T linalg.Number] struct {
	Feature       []T
	Neighbors     uint
	MaxCandidates uint
}

func createIndex[T linalg.Number](ctx context.Context, features [][]T, ind string, nDim uint, leafSize uint, nTrees uint) (countrymaam.Index[T], error) {
	switch ind {
	case "flat":
		builder := index.NewFlatIndexBuilder[T](nDim)
		return builder.Build(ctx, features)
	case "kd-tree":
		kdTreeBuilder := bsp_tree.NewKdTreeBuilder[T]()
		kdTreeBuilder.SetLeafs(leafSize)
		builder := index.NewBspTreeIndexBuilder[T](nDim, kdTreeBuilder)
		return builder.Build(ctx, features)
	case "rkd-tree":
		kdTreeBuilder := bsp_tree.NewKdTreeBuilder[T]()
		kdTreeBuilder.SetLeafs(leafSize).SetSampleFeatures(100).SetTopKCandidates(5)
		builder := index.NewBspTreeIndexBuilder[T](nDim, kdTreeBuilder)
		builder.SetTrees(nTrees)
		return builder.Build(ctx, features)
	case "rp-tree":
		rpTreeBuilder := bsp_tree.NewRpTreeBuilder[T]()
		rpTreeBuilder.SetLeafs(leafSize)
		builder := index.NewBspTreeIndexBuilder[T](nDim, rpTreeBuilder)
		return builder.Build(ctx, features)
	case "rrp-tree":
		rpTreeBuilder := bsp_tree.NewRpTreeBuilder[T]()
		rpTreeBuilder.SetLeafs(leafSize).SetSampleFeatures(32)
		builder := index.NewBspTreeIndexBuilder[T](nDim, rpTreeBuilder)
		builder.SetTrees(nTrees)
		return builder.Build(ctx, features)
	case "aknn":
		graphBuilder := graph.NewAKnnGraphBuilder[T]()
		graphBuilder.SetK(30).SetRho(1.0)

		builder := index.NewGraphIndexBuilder[T](nDim, graphBuilder)
		return builder.Build(ctx, features)
	case "rpaknn":
		rpTreeBuilder := bsp_tree.NewRpTreeBuilder[T]()
		rpTreeBuilder.SetLeafs(leafSize)
		rpBuilder := index.NewBspTreeIndexBuilder[T](nDim, rpTreeBuilder)
		rpBuilder.SetTrees(1)

		graphBuilder := graph.NewAKnnGraphBuilder[T]()
		graphBuilder.SetK(30).SetRho(1.0)
		aknnBuilder := index.NewGraphIndexBuilder[T](nDim, graphBuilder)

		builder := index.NewCompositeIndexBuilder[T, index.BspTreeIndex[T], index.GraphIndex[T]](rpBuilder, aknnBuilder)
		builder.SetEntriesNum(32)
		return builder.Build(context.Background(), features)
	default:
		return nil, fmt.Errorf("unknown index name: %s", ind)
	}
}

func loadIndex[T linalg.Number](ind string, inputPath string) (countrymaam.Index[T], error) {
	file, err := os.Open(inputPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	switch ind {
	case "flat":
		return index.LoadFlatIndex[T](file)
	case "kd-tree":
		return index.LoadBspTreeIndex[T](file)
	case "rkd-tree":
		return index.LoadBspTreeIndex[T](file)
	case "rp-tree":
		return index.LoadBspTreeIndex[T](file)
	case "rrp-tree":
		return index.LoadBspTreeIndex[T](file)
	case "aknn":
		return index.LoadGraphIndex[T](file)
	case "rpaknn":
		return index.LoadCompositeIndex[T](file)
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

	log.Println("reading data...")
	r := bufio.NewReader(os.Stdin)
	features := make([][]T, 0, 100000)
Loop:
	for i := 0; ; i++ {
		feature, err := readFeature[T](r, nDim)
		if err == io.EOF {
			break Loop
		}
		if err != nil {
			return err
		}

		features = append(features, feature)
	}
	log.Println("done")

	log.Println("building index...")
	ctx := context.Background()
	index, err := createIndex(ctx, features, indexName, nDim, leafSize, nTrees)
	if err != nil {
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
	sockPath := c.String("sock")

	r := bufio.NewReader(os.Stdin)
	w := bufio.NewWriter(os.Stdout)

	if sockPath != "" {
		os.Remove(sockPath)

		listener, err := net.Listen("unix", sockPath)
		if err != nil {
			return err
		}

		conn, err := listener.Accept()
		if err != nil {
			return err
		}
		defer conn.Close()

		r = bufio.NewReader(conn)
		w = bufio.NewWriter(conn)
	}

	log.Println("start predictAction")

	switch dtype {
	case "float32":
		return predict[float32](nDim, indexName, inputName, profileOutputName, r, w)
	case "uint8":
		return predict[uint8](nDim, indexName, inputName, profileOutputName, r, w)
	default:
		return fmt.Errorf("unknown dtype: %s", dtype)
	}

}

func predict[T linalg.Number](nDim uint, indexName string, inputName string, profileOutputName string, r *bufio.Reader, w *bufio.Writer) error {
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
	index, err := loadIndex[T](indexName, inputName)
	if err != nil {
		return err
	}

Loop:
	for {
		query, err := readQuery[T](r, nDim)
		if err == io.EOF {
			break Loop
		}
		if err != nil {
			return err
		}

		ch := index.SearchChannel(ctx, query.Feature)
		neighbors, err := countrymaam.Search(ch, query.Neighbors, query.MaxCandidates)
		if err != nil {
			return err
		}

		if err := binary.Write(w, binary.LittleEndian, uint32(len(neighbors))); err != nil {
			return err
		}
		for _, n := range neighbors {
			if err := binary.Write(w, binary.LittleEndian, uint32(n.Index)); err != nil {
				return err
			}
		}
		w.Flush()
	}

	if true {
		log.Println("query ok")
		return errors.New("query ok")
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
					&cli.StringFlag{
						Name:  "sock",
						Value: "",
						Usage: "domain socket path",
					},
				},
			},
		},
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal(err)
	}
}
