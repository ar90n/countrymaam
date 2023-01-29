package linalg

import (
	"fmt"
	"testing"

	"github.com/ar90n/countrymaam/linalg/asm"
	"github.com/stretchr/testify/assert"
)

func Test_DotF32(t *testing.T) {
	type TestCase struct {
		Name string
		X, Y []float32
		Want float32
		F    func(x, y []float32) float32
		Skip bool
	}

	testCases := []TestCase{
		{
			Name: "vanilla",
			X:    []float32{15, 6, 1, 12, 13, 9, 15, 18, 14, 6, 17, 16, 1, 3, 3, 9, 5, 13, 6, 20},
			Y:    []float32{9, 2, 18, 13, 10, 1, 7, 10, 8, 7, 11, 18, 5, 19, 17, 19, 15, 0, 17, 18},
			Want: 2195,
			F:    dot[float32, float32],
			Skip: false,
		},
		{
			Name: "avx2",
			X:    []float32{15, 6, 1, 12, 13, 9, 15, 18, 14, 6, 17, 16, 1, 3, 3, 9, 5, 13, 6, 20},
			Y:    []float32{9, 2, 18, 13, 10, 1, 7, 10, 8, 7, 11, 18, 5, 19, 17, 19, 15, 0, 17, 18},
			Want: 2195,
			F:    asm.DotF32AVX2,
			Skip: false,
		},
	}

	for _, tc := range testCases {
		if tc.Skip {
			continue
		}
		assert.InEpsilon(t, tc.Want, tc.F(tc.X, tc.Y), 0.0001)
	}

	//e := GenEngine(l)
	//if cpu.X86.HasAVX2 {
	//	fmt.Println(e)
	//}
}

func Test_DotU8(t *testing.T) {
	type TestCase struct {
		Name string
		X, Y []uint8
		Want float32
		F    func(x, y []uint8) float32
		Skip bool
	}

	testCases := []TestCase{
		{
			Name: "vanilla",
			X:    []uint8{15, 6, 1, 12, 13, 9, 15, 18, 14, 6, 17, 16, 1, 3, 3, 9, 5, 13, 6, 20},
			Y:    []uint8{9, 2, 18, 13, 10, 1, 7, 10, 8, 7, 11, 18, 5, 19, 17, 19, 15, 0, 17, 18},
			Want: 2195,
			F:    dot[uint8, uint8],
			Skip: false,
		},
	}

	for _, tc := range testCases {
		if tc.Skip {
			continue
		}
		assert.InEpsilon(t, tc.Want, tc.F(tc.X, tc.Y), 0.0001)
	}
}

func Test_SqDistF32(t *testing.T) {
	type TestCase struct {
		Name string
		X, Y []float32
		Want float32
		F    func(x, y []float32) float32
		Skip bool
	}

	testCases := []TestCase{
		{
			Name: "vanilla",
			X:    []float32{15, 6, 1, 12, 13, 9, 15, 18, 14, 6, 17, 16, 1, 3, 3, 9, 5, 13, 6, 20},
			Y:    []float32{9, 2, 18, 13, 10, 1, 7, 10, 8, 7, 11, 18, 5, 19, 17, 19, 15, 0, 17, 18},
			Want: 1582,
			F:    sqL2[float32, float32],
			Skip: false,
		},
		{
			Name: "avx",
			X:    []float32{15, 6, 1, 12, 13, 9, 15, 18, 14, 6, 17, 16, 1, 3, 3, 9, 5, 13, 6, 20},
			Y:    []float32{9, 2, 18, 13, 10, 1, 7, 10, 8, 7, 11, 18, 5, 19, 17, 19, 15, 0, 17, 18},
			Want: 1582,
			F:    asm.SqL2F32AVX2,
			Skip: false,
		},
	}

	for _, tc := range testCases {
		if tc.Skip {
			continue
		}
		assert.InEpsilon(t, tc.Want, tc.F(tc.X, tc.Y), 0.0001)
	}
}

func Test_SqDistU8(t *testing.T) {
	type TestCase struct {
		Name string
		X, Y []uint8
		Want float32
		F    func(x, y []uint8) float32
		Skip bool
	}

	testCases := []TestCase{
		{
			Name: "vanilla",
			X:    []uint8{15, 6, 1, 12, 13, 9, 15, 18, 14, 6, 17, 16, 1, 3, 3, 9, 5, 13, 6, 20},
			Y:    []uint8{9, 2, 18, 13, 10, 1, 7, 10, 8, 7, 11, 18, 5, 19, 17, 19, 15, 0, 17, 18},
			Want: 1582,
			F:    sqL2[uint8, uint8],
			Skip: false,
		},
	}

	for _, tc := range testCases {
		if tc.Skip {
			continue
		}
		assert.InEpsilon(t, tc.Want, tc.F(tc.X, tc.Y), 0.0001)
	}
}

type TestNewLinAlgTestCase struct {
	Name    string
	Options LinAlgOptions
	Skip    bool
}

func testNewLinAlgImpl[T Number](t *testing.T, f func(LinAlgOptions) Env[T], suffix string) {

	for _, tc := range []TestNewLinAlgTestCase{
		{
			Name:    fmt.Sprintf("vanilla_%s", suffix),
			Options: LinAlgOptions{},
			Skip:    false,
		},
		{
			Name: fmt.Sprintf("avx2_%s", suffix),
			Options: LinAlgOptions{
				UseAVX2: true,
			},
			Skip: true,
		},
	} {
		if tc.Skip {
			t.Skip(tc.Name)
			continue
		}

		f(tc.Options)
	}
}

func Test_NewLinAlg(t *testing.T) {
	testNewLinAlgImpl(t, NewLinAlg[uint8], "u8")
	testNewLinAlgImpl(t, NewLinAlg[float32], "f32")
}
