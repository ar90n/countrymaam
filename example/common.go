package main

import (
	"bytes"
	_ "embed"
	"encoding/csv"
	"io"
	"strconv"
)

//go:embed dim064.csv
var dim64 []byte

func readFeatures(dim uint) ([][]uint8, error) {
	r := csv.NewReader(bytes.NewReader(dim64))

	ret := make([][]uint8, 0)
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		feature := make([]uint8, dim)
		for i, text := range record {
			val, err := strconv.ParseInt(text, 10, 32)
			if err != nil {
				return nil, err
			}
			feature[i] = uint8(val)
		}
		ret = append(ret, feature)
	}

	return ret, nil
}

func readFeatures2(dim uint) ([][]float32, error) {
	r := csv.NewReader(bytes.NewReader(dim64))

	ret := make([][]float32, 0)
	for {
		record, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}

		feature := make([]float32, dim)
		for i, text := range record {
			val, err := strconv.ParseInt(text, 10, 32)
			if err != nil {
				return nil, err
			}
			feature[i] = float32(val)
		}
		ret = append(ret, feature)
	}

	return ret, nil
}
