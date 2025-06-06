; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=ve -mattr=+vpu | FileCheck %s

declare <512 x i32> @llvm.vp.add.v512i32(<512 x i32>, <512 x i32>, <512 x i1>, i32)

define fastcc <512 x i32> @test_vp_add_v512i32_vv(<512 x i32> %i0, <512 x i32> %i1, <512 x i1> %m, i32 %n) {
; CHECK-LABEL: test_vp_add_v512i32_vv:
; CHECK:       # %bb.0:
; CHECK-NEXT:    adds.w.sx %s0, 1, %s0
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    srl %s0, %s0, 1
; CHECK-NEXT:    lvl %s0
; CHECK-NEXT:    pvaddu %v0, %v0, %v1, %vm2
; CHECK-NEXT:    b.l.t (, %s10)
  %r0 = call <512 x i32> @llvm.vp.add.v512i32(<512 x i32> %i0, <512 x i32> %i1, <512 x i1> %m, i32 %n)
  ret <512 x i32> %r0
}

define fastcc <512 x i32> @test_vp_add_v512i32_rv(i32 %s0, <512 x i32> %i1, <512 x i1> %m, i32 %n) {
; CHECK-LABEL: test_vp_add_v512i32_rv:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    sll %s2, %s0, 32
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    or %s0, %s0, %s2
; CHECK-NEXT:    adds.w.sx %s1, 1, %s1
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    srl %s1, %s1, 1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvaddu %v0, %s0, %v0, %vm2
; CHECK-NEXT:    b.l.t (, %s10)
  %xins = insertelement <512 x i32> undef, i32 %s0, i32 0
  %i0 = shufflevector <512 x i32> %xins, <512 x i32> undef, <512 x i32> zeroinitializer
  %r0 = call <512 x i32> @llvm.vp.add.v512i32(<512 x i32> %i0, <512 x i32> %i1, <512 x i1> %m, i32 %n)
  ret <512 x i32> %r0
}

define fastcc <512 x i32> @test_vp_add_v512i32_vr(<512 x i32> %i0, i32 %s1, <512 x i1> %m, i32 %n) {
; CHECK-LABEL: test_vp_add_v512i32_vr:
; CHECK:       # %bb.0:
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    sll %s2, %s0, 32
; CHECK-NEXT:    and %s0, %s0, (32)0
; CHECK-NEXT:    or %s0, %s0, %s2
; CHECK-NEXT:    adds.w.sx %s1, 1, %s1
; CHECK-NEXT:    and %s1, %s1, (32)0
; CHECK-NEXT:    srl %s1, %s1, 1
; CHECK-NEXT:    lvl %s1
; CHECK-NEXT:    pvaddu %v0, %s0, %v0, %vm2
; CHECK-NEXT:    b.l.t (, %s10)
  %yins = insertelement <512 x i32> undef, i32 %s1, i32 0
  %i1 = shufflevector <512 x i32> %yins, <512 x i32> undef, <512 x i32> zeroinitializer
  %r0 = call <512 x i32> @llvm.vp.add.v512i32(<512 x i32> %i0, <512 x i32> %i1, <512 x i1> %m, i32 %n)
  ret <512 x i32> %r0
}
