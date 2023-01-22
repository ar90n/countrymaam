package countrymaam

import (
	"bytes"
	"encoding/gob"
	"io"
)

func saveIndex[T any](index *T, w io.Writer) error {
	var buffer bytes.Buffer
	enc := gob.NewEncoder(&buffer)
	if err := enc.Encode(index); err != nil {
		return err
	}

	beg := 0
	byteArray := buffer.Bytes()
	for beg < len(byteArray) {
		n, err := w.Write(byteArray[beg:])
		if err != nil {
			return err
		}
		beg += n
	}

	return nil
}

func loadIndex[T any](r io.Reader) (ret T, _ error) {
	dec := gob.NewDecoder(r)
	if err := dec.Decode(&ret); err != nil {
		return ret, err
	}

	return ret, nil
}
