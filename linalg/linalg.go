package linalg

import (
	"context"
	"reflect"

	"github.com/ar90n/countrymaam/linalg/asm"
	"golang.org/x/sys/cpu"
)

const linAlgKey = "linAlg"

type Env[T Number] struct {
	SqL2        func(x, y []T) float32
	SqL2WithF32 func(x []T, y []float32) float32
	Dot         func(x, y []T) float32
	DotWithF32  func(x []T, y []float32) float32
}

type Config struct {
	DisableAVX2 bool
}

func WithLinAlg(ctx context.Context, conf Config) context.Context {
	return context.WithValue(ctx, linAlgKey, conf)
}

func NewLinAlgFromContext[T Number](ctx context.Context) Env[T] {
	if conf := ctx.Value(linAlgKey); conf != nil {
		return NewLinAlg[T](conf.(Config))
	}

	return NewLinAlg[T](Config{})
}

func NewLinAlg[T Number](conf Config) Env[T] {
	if reflect.ValueOf(*new(T)).Kind() == reflect.Float32 {
		return newLinAlgF32(conf).(Env[T])
	}

	if reflect.ValueOf(*new(T)).Kind() == reflect.Uint8 {
		return newLinAlgUint8(conf).(Env[T])
	}

	return Env[T]{
		SqL2:        sqL2[T, T],
		SqL2WithF32: sqL2[T, float32],
		Dot:         dot[T, T],
		DotWithF32:  dot[T, float32],
	}
}

func newLinAlgF32(conf Config) interface{} {
	if cpu.X86.HasAVX2 && !conf.DisableAVX2 {
		return Env[float32]{
			SqL2:        asm.SqL2F32AVX2,
			SqL2WithF32: asm.SqL2F32AVX2,
			Dot:         asm.DotF32AVX2,
			DotWithF32:  asm.DotF32AVX2,
		}
	}

	return Env[float32]{
		SqL2:        sqL2[float32, float32],
		SqL2WithF32: sqL2[float32, float32],
		Dot:         dot[float32, float32],
		DotWithF32:  dot[float32, float32],
	}
}

func newLinAlgUint8(options Config) interface{} {
	if cpu.X86.HasAVX2 && !options.DisableAVX2 {
		return Env[uint8]{
			//SqL2:        asm.SqL2Uint8AVX2,
			SqL2:        sqL2[uint8, uint8],
			SqL2WithF32: sqL2[uint8, float32],
			//Dot:         asm.DotUint8AVX2,
			Dot:        dot[uint8, uint8],
			DotWithF32: dot[uint8, float32],
		}
	}

	return Env[uint8]{
		SqL2:        sqL2[uint8, uint8],
		SqL2WithF32: sqL2[uint8, float32],
		Dot:         dot[uint8, uint8],
		DotWithF32:  dot[uint8, float32],
	}
}

// vanilla implementations

func sqL2[T Number, U Number](x []T, y []U) float32 {
	dist := float32(0.0)
	for i := 0; i < len(x); i++ {
		diff := float32(x[i]) - float32(y[i])
		dist += diff * diff
	}

	return dist
}

func dot[T Number, U Number](x []T, y []U) float32 {
	dot := float32(0.0)
	for i := 0; i < len(x); i++ {
		dot += float32(x[i]) * float32(y[i])
	}

	return dot
}
