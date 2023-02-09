package pipeline

import (
	"context"
)

const streamBufferSize = 8

func Take[T any](ctx context.Context, n uint, inputStream <-chan T) <-chan T {
	outputStream := make(chan T, streamBufferSize)
	go func() {
		defer close(outputStream)

		i := uint(0)
		for {
			select {
			case <-ctx.Done():
				return
			case item, ok := <-inputStream:
				if !ok {
					return
				}
				outputStream <- item
				i++
				if n <= i {
					return
				}
			}
		}
	}()

	return outputStream
}

func Unique[T comparable](ctx context.Context, inputStream <-chan T) <-chan T {
	outputStream := make(chan T, streamBufferSize)
	go func() {
		defer close(outputStream)

		founds := map[T]interface{}{}
		for {
			select {
			case <-ctx.Done():
				return
			case item, ok := <-inputStream:
				if !ok {
					return
				}

				if _, ok := founds[item]; ok {
					continue
				}
				founds[item] = struct{}{}

				outputStream <- item
			}
		}
	}()

	return outputStream
}

func Seq(ctx context.Context, n uint) <-chan int {
	outputStream := make(chan int, streamBufferSize)
	go func() {
		defer close(outputStream)
		for i := uint(0); i < n; i++ {
			select {
			case <-ctx.Done():
				return
			case outputStream <- int(i):
			}
		}
	}()

	return outputStream
}

func ToSlice[T any](ctx context.Context, inputStream <-chan T) []T {
	output := make([]T, 0)
	for item := range inputStream {
		output = append(output, item)
	}

	return output
}

func OrDone[T any](ctx context.Context, inputStream <-chan T) <-chan T {
	outputStream := make(chan T, streamBufferSize)
	go func() {
		defer close(outputStream)
		for {
			select {
			case <-ctx.Done():
				return
			case v, ok := <-inputStream:
				if !ok {
					return
				}

				select {
				case <-ctx.Done():
				case outputStream <- v:
				}
			}
		}
	}()

	return outputStream
}
