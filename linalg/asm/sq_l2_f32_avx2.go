//go:generate go run ./sq_l2_f32_avx2.go -out sq_l2_f32_avx2.s -stubs sq_l2_f32_stub_avx2.go
//go:build ignore
// +build ignore

package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

var unroll_sq_l2_f32_avx2 = 1

func main() {
	TEXT("SqL2F32AVX2", NOSPLIT, "func(x, y []float32) float32")
	x := Mem{Base: Load(Param("x").Base(), GP64())}
	y := Mem{Base: Load(Param("y").Base(), GP64())}
	n := Load(Param("x").Len(), GP64())

	// Allocate accumulation registers.
	acc := make([]VecVirtual, unroll_sq_l2_f32_avx2)
	for i := 0; i < unroll_sq_l2_f32_avx2; i++ {
		acc[i] = YMM()
	}

	// Zero initialization.
	for i := 0; i < unroll_sq_l2_f32_avx2; i++ {
		VXORPS(acc[i], acc[i], acc[i])
	}

	// Loop over blocks and process them with vector instructions.
	blockitems := 8 * unroll_sq_l2_f32_avx2
	blocksize := 4 * blockitems
	Label("blockloop")
	CMPQ(n, U32(blockitems))
	JL(LabelRef("tail"))

	// Load x.
	xs := make([]VecVirtual, unroll_sq_l2_f32_avx2)
	for i := 0; i < unroll_sq_l2_f32_avx2; i++ {
		xs[i] = YMM()
	}

	for i := 0; i < unroll_sq_l2_f32_avx2; i++ {
		VMOVUPS(x.Offset(32*i), xs[i])
	}

	// The actual square of difference.
	for i := 0; i < unroll_sq_l2_f32_avx2; i++ {
		VSUBPS(y.Offset(32*i), xs[i], xs[i])
		VFMADD231PS(xs[i], xs[i], acc[i])
	}

	ADDQ(U32(blocksize), x.Base)
	ADDQ(U32(blocksize), y.Base)
	SUBQ(U32(blockitems), n)
	JMP(LabelRef("blockloop"))

	// Process any trailing entries.
	Label("tail")
	tail := XMM()
	VXORPS(tail, tail, tail)

	Label("tailloop")
	CMPQ(n, U32(0))
	JE(LabelRef("reduce"))

	xt := XMM()
	VMOVSS(x, xt)
	SUBSS(y, xt)
	VFMADD231SS(xt, xt, tail)

	ADDQ(U32(4), x.Base)
	ADDQ(U32(4), y.Base)
	DECQ(n)
	JMP(LabelRef("tailloop"))

	// Reduce the lanes to one.
	Label("reduce")
	for i := 1; i < unroll_sq_l2_f32_avx2; i++ {
		VADDPS(acc[0], acc[i], acc[0])
	}

	result := acc[0].AsX()
	top := XMM()
	VEXTRACTF128(U8(1), acc[0], top)
	VADDPS(result, top, result)
	VADDPS(result, tail, result)
	VHADDPS(result, result, result)
	VHADDPS(result, result, result)
	Store(result, ReturnIndex(0))

	RET()

	Generate()
}
