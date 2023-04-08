package index

import (
	"context"
	"encoding/gob"
	"fmt"
	"io"

	"github.com/ar90n/countrymaam"
	"github.com/ar90n/countrymaam/bsp_tree"
	"github.com/ar90n/countrymaam/linalg"
	"github.com/ar90n/countrymaam/pipeline"
)

type CompositeIndex[T linalg.Number] struct {
	HeadIndex  countrymaam.Index[T]
	TailIndex  countrymaam.EntryPointIndex[T]
	EntriesNum uint
}

func (ci CompositeIndex[T]) SearchChannel(ctx context.Context, query []T) <-chan countrymaam.SearchResult {
	outputStream := make(chan countrymaam.SearchResult, streamBufferSize)

	go func() {
		defer close(outputStream)

		entries := []uint{}
		entriesCh := ci.HeadIndex.SearchChannel(ctx, query)
		for ret := range pipeline.OrDone(ctx, entriesCh) {
			entries = append(entries, ret.Index)
			if ci.EntriesNum <= uint(len(entries)) {
				break
			}
		}

		searchCh := ci.TailIndex.SearchChannelWithEntries(ctx, query, entries)
		for ret := range pipeline.OrDone(ctx, searchCh) {
			outputStream <- ret
		}
	}()

	return outputStream
}

func (ci CompositeIndex[T]) Save(w io.Writer) error {
	return saveIndex(ci, w)
}

type CompositeIndexBuilder[T linalg.Number, HI countrymaam.Index[T], TI countrymaam.EntryPointIndex[T]] struct {
	HeadIndexBuilder countrymaam.IndexBuilder[T, HI]
	TailIndexBuilder countrymaam.IndexBuilder[T, TI]
	EntriesNum       uint
}

func NewCompositeIndexBuilder[T linalg.Number, HI countrymaam.Index[T], TI countrymaam.EntryPointIndex[T]](headIndexBuilder countrymaam.IndexBuilder[T, HI], tailIndexBuilder countrymaam.IndexBuilder[T, TI]) CompositeIndexBuilder[T, HI, TI] {
	return CompositeIndexBuilder[T, HI, TI]{
		HeadIndexBuilder: headIndexBuilder,
		TailIndexBuilder: tailIndexBuilder,
		EntriesNum:       1,
	}
}

func (cib *CompositeIndexBuilder[T, HI, TI]) SetEntriesNum(entriesNum uint) {
	cib.EntriesNum = entriesNum
}

func (cib *CompositeIndexBuilder[T, HI, TI]) GetPrameterString() string {
	return fmt.Sprintf("%s_%s", cib.HeadIndexBuilder.GetPrameterString(), cib.TailIndexBuilder.GetPrameterString())
}

func (cib CompositeIndexBuilder[T, HI, TI]) Build(ctx context.Context, features [][]T) (*CompositeIndex[T], error) {
	headIndex, err := cib.HeadIndexBuilder.Build(ctx, features)
	if err != nil {
		return nil, err
	}

	tailIndex, err := cib.TailIndexBuilder.Build(ctx, features)
	if err != nil {
		return nil, err
	}

	index := &CompositeIndex[T]{
		HeadIndex:  *headIndex,
		TailIndex:  *tailIndex,
		EntriesNum: cib.EntriesNum,
	}
	return index, nil
}

func LoadCompositeIndex[T linalg.Number](r io.Reader) (*CompositeIndex[T], error) {
	gob.Register(CompositeIndex[T]{})
	gob.Register(BspTreeIndex[T]{})
	gob.Register(GraphIndex[T]{})
	bsp_tree.Register[T]()

	index, err := loadIndex[CompositeIndex[T]](r)
	if err != nil {
		return nil, err
	}

	return &index, nil
}
