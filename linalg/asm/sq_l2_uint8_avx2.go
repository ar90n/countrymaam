//go:generate go run ./sq_l2_uint8_avx2.go -out sql2_uint8_avx2.s -stubs sq_l2_uint8_stub_avx2.go
//go:build ignore
// +build ignore

package main

import (
	. "github.com/mmcloughlin/avo/build"
	. "github.com/mmcloughlin/avo/operand"
	. "github.com/mmcloughlin/avo/reg"
)

var unroll_sql2_uint8_avx2 = 1

func main() {
	TEXT("SqL2Uint8AVX2", NOSPLIT, "func(x, y []uint8) float32")
	x := Mem{Base: Load(Param("x").Base(), GP64())}
	y := Mem{Base: Load(Param("y").Base(), GP64())}
	n := Load(Param("x").Len(), GP64())
	one := ConstData("one", U16(0x0001))

	// Allocate accumulation registers.
	acc := make([]VecVirtual, unroll_sql2_uint8_avx2)
	for i := 0; i < unroll_sql2_uint8_avx2; i++ {
		acc[i] = YMM()
	}

	// Zero initialization.
	for i := 0; i < unroll_sql2_uint8_avx2; i++ {
		VXORPS(acc[i], acc[i], acc[i])
	}

	// Loop over blocks and process them with vector instructions.
	blockitems := 32 * unroll_sql2_uint8_avx2
	blocksize := 1 * blockitems
	Label("blockloop")
	CMPQ(n, U32(blockitems))
	JL(LabelRef("tail"))

	xs := make([]VecVirtual, unroll_sql2_uint8_avx2)
	ys := make([]VecVirtual, unroll_sql2_uint8_avx2)
	for i := 0; i < unroll_sql2_uint8_avx2; i++ {
		xs[i] = YMM()
		ys[i] = YMM()
	}

	for i := 0; i < unroll_sql2_uint8_avx2; i++ {
		VMOVDQU(x.Offset(32*i), xs[i])
		VMOVDQU(y.Offset(32*i), ys[i])
	}

	ones := YMM()
	VPBROADCASTW(one, ones)
	for i := 0; i < unroll_sql2_uint8_avx2; i++ {
		maxv := YMM()
		minv := YMM()
		VPMAXUB(xs[i], ys[i], maxv)
		VPMINUB(xs[i], ys[i], minv)
		VPSUBUSB(minv, maxv, xs[i])
		VPMADDUBSW(xs[i], xs[i], xs[i])
		VPMADDWD(xs[i], ones, xs[i])
		VCVTDQ2PS(xs[i], xs[i])
		VPADDUSB(xs[i], acc[i], acc[i])
	}

	ADDQ(U32(blocksize), x.Base)
	ADDQ(U32(blocksize), y.Base)
	SUBQ(U32(blockitems), n)
	JMP(LabelRef("blockloop"))

	// Process any trailing entries.
	Label("tail")
	tail := XMM()
	xt := GP64()
	yt := GP64()
	m := XMM()
	VXORPS(tail, tail, tail)
	XORQ(xt, xt)
	XORQ(yt, yt)

	Label("tailloop")
	CMPQ(n, U32(0))
	JE(LabelRef("reduce"))

	MOVB(x, xt.As8())
	MOVB(y, yt.As8())
	SUBQ(yt, xt)
	IMULQ(xt, xt)
	MOVQ(xt, m)
	VCVTDQ2PS(m, m)
	ADDPS(m, tail)

	ADDQ(U32(1), x.Base)
	ADDQ(U32(1), y.Base)
	MOVD(U32(0), xt)
	DECQ(n)

	JMP(LabelRef("tailloop"))

	// Reduce the lanes to one.
	Label("reduce")
	for i := 1; i < unroll_sql2_uint8_avx2; i++ {
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
