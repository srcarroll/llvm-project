; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc -O0 -mtriple=riscv32 -mattr=+xandesperf -verify-machineinstrs < %s \
; RUN:   | FileCheck %s

; NDS.BFOZ

; MSB >= LSB

define i32 @bfoz_from_and_i32(i32 %x) {
; CHECK-LABEL: bfoz_from_and_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfoz a0, a0, 11, 0
; CHECK-NEXT:    ret
  %a = and i32 %x, 4095
  ret i32 %a
}

define i64 @bfoz_from_and_i64(i64 %x) {
; CHECK-LABEL: bfoz_from_and_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    nds.bfoz a0, a0, 11, 0
; CHECK-NEXT:    li a1, 0
; CHECK-NEXT:    ret
  %a = and i64 %x, 4095
  ret i64 %a
}

define i32 @bfoz_from_and_lshr_i32(i32 %x) {
; CHECK-LABEL: bfoz_from_and_lshr_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfoz a0, a0, 25, 23
; CHECK-NEXT:    ret
  %shifted = lshr i32 %x, 23
  %masked = and i32 %shifted, 7
  ret i32 %masked
}

define i64 @bfoz_from_and_lshr_i64(i64 %x) {
; CHECK-LABEL: bfoz_from_and_lshr_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x12 killed $x11
; CHECK-NEXT:    nds.bfoz a0, a1, 25, 14
; CHECK-NEXT:    li a1, 0
; CHECK-NEXT:    ret
  %shifted = lshr i64 %x, 46
  %masked = and i64 %shifted, 4095
  ret i64 %masked
}

define i32 @bfoz_from_lshr_and_i32(i32 %x) {
; CHECK-LABEL: bfoz_from_lshr_and_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfoz a0, a0, 23, 12
; CHECK-NEXT:    ret
  %masked = and i32 %x, 16773120
  %shifted = lshr i32 %masked, 12
  ret i32 %shifted
}

define i64 @bfoz_from_lshr_and_i64(i64 %x) {
; CHECK-LABEL: bfoz_from_lshr_and_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x12 killed $x11
; CHECK-NEXT:    # kill: def $x12 killed $x10
; CHECK-NEXT:    andi a1, a1, 15
; CHECK-NEXT:    srli a0, a0, 24
; CHECK-NEXT:    slli a1, a1, 8
; CHECK-NEXT:    or a0, a0, a1
; CHECK-NEXT:    li a1, 0
; CHECK-NEXT:    ret
  %masked = and i64 %x, 68702699520
  %shifted = lshr i64 %masked, 24
  ret i64 %shifted
}

; MSB = 0

define i32 @bfoz_from_and_shl_with_msb_zero_i32(i32 %x) {
; CHECK-LABEL: bfoz_from_and_shl_with_msb_zero_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfoz a0, a0, 0, 15
; CHECK-NEXT:    ret
  %shifted = shl i32 %x, 15
  %masked = and i32 %shifted, 32768
  ret i32 %masked
}

define i32 @bfoz_from_lshr_shl_with_msb_zero_i32(i32 %x) {
; CHECK-LABEL: bfoz_from_lshr_shl_with_msb_zero_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfoz a0, a0, 0, 18
; CHECK-NEXT:    ret
  %shl = shl i32 %x, 31
  %lshr = lshr i32 %shl, 13
  ret i32 %lshr
}

; MSB < LSB

define i32 @bfoz_from_and_shl_i32(i32 %x) {
; CHECK-LABEL: bfoz_from_and_shl_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfoz a0, a0, 12, 23
; CHECK-NEXT:    ret
  %shifted = shl i32 %x, 12
  %masked = and i32 %shifted, 16773120
  ret i32 %masked
}

define i32 @bfoz_from_lshr_shl_i32(i32 %x) {
; CHECK-LABEL: bfoz_from_lshr_shl_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfoz a0, a0, 19, 24
; CHECK-NEXT:    ret
  %shl = shl i32 %x, 26
  %lshr = lshr i32 %shl, 7
  ret i32 %lshr
}

; NDS.BFOS

; MSB >= LSB

define i32 @bfos_from_ashr_shl_i32(i32 %x) {
; CHECK-LABEL: bfos_from_ashr_shl_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfos a0, a0, 23, 16
; CHECK-NEXT:    ret
  %shl = shl i32 %x, 8
  %ashr = ashr i32 %shl, 24
  ret i32 %ashr
}

define i32 @bfos_from_ashr_sexti8_i32(i8 %x) {
; CHECK-LABEL: bfos_from_ashr_sexti8_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    nds.bfos a0, a0, 7, 5
; CHECK-NEXT:    ret
  %sext = sext i8 %x to i32
  %ashr = ashr i32 %sext, 5
  ret i32 %ashr
}

define i32 @bfos_from_ashr_sexti16_i32(i16 %x) {
; CHECK-LABEL: bfos_from_ashr_sexti16_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    nds.bfos a0, a0, 15, 11
; CHECK-NEXT:    ret
  %sext = sext i16 %x to i32
  %ashr = ashr i32 %sext, 11
  ret i32 %ashr
}

; MSB = 0

define i32 @bfos_from_ashr_shl_with_msb_zero_insert_i32(i32 %x) {
; CHECK-LABEL: bfos_from_ashr_shl_with_msb_zero_insert_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfos a0, a0, 0, 14
; CHECK-NEXT:    ret
  %shl = shl i32 %x, 31
  %lshr = ashr i32 %shl, 17
  ret i32 %lshr
}

; MSB < LSB

define i32 @bfos_from_ashr_shl_insert_i32(i32 %x) {
; CHECK-LABEL: bfos_from_ashr_shl_insert_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfos a0, a0, 18, 20
; CHECK-NEXT:    ret
  %shl = shl i32 %x, 29
  %lshr = ashr i32 %shl, 11
  ret i32 %lshr
}

; sext

define i32 @sexti1_i32(i32 %a) {
; CHECK-LABEL: sexti1_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfos a0, a0, 0, 0
; CHECK-NEXT:    ret
  %shl = shl i32 %a, 31
  %shr = ashr exact i32 %shl, 31
  ret i32 %shr
}

define i32 @sexti1_i32_2(i1 %a) {
; CHECK-LABEL: sexti1_i32_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    nds.bfos a0, a0, 0, 0
; CHECK-NEXT:    ret
  %1 = sext i1 %a to i32
  ret i32 %1
}

define i32 @sexti8_i32(i32 %a) {
; CHECK-LABEL: sexti8_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfos a0, a0, 7, 0
; CHECK-NEXT:    ret
  %shl = shl i32 %a, 24
  %shr = ashr exact i32 %shl, 24
  ret i32 %shr
}

define i32 @sexti8_i32_2(i8 %a) {
; CHECK-LABEL: sexti8_i32_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    nds.bfos a0, a0, 7, 0
; CHECK-NEXT:    ret
  %1 = sext i8 %a to i32
  ret i32 %1
}

define i32 @sexti16_i32(i32 %a) {
; CHECK-LABEL: sexti16_i32:
; CHECK:       # %bb.0:
; CHECK-NEXT:    nds.bfos a0, a0, 15, 0
; CHECK-NEXT:    ret
  %shl = shl i32 %a, 16
  %shr = ashr exact i32 %shl, 16
  ret i32 %shr
}

define i32 @sexti16_i32_2(i16 %a) {
; CHECK-LABEL: sexti16_i32_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    nds.bfos a0, a0, 15, 0
; CHECK-NEXT:    ret
  %1 = sext i16 %a to i32
  ret i32 %1
}

define i64 @sexti1_i64(i64 %a) {
; CHECK-LABEL: sexti1_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    nds.bfos a1, a0, 0, 0
; CHECK-NEXT:    mv a0, a1
; CHECK-NEXT:    ret
  %shl = shl i64 %a, 63
  %shr = ashr exact i64 %shl, 63
  ret i64 %shr
}

define i64 @sexti1_i64_2(i1 %a) {
; CHECK-LABEL: sexti1_i64_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    nds.bfos a1, a0, 0, 0
; CHECK-NEXT:    mv a0, a1
; CHECK-NEXT:    ret
  %1 = sext i1 %a to i64
  ret i64 %1
}

define i64 @sexti8_i64(i64 %a) {
; CHECK-LABEL: sexti8_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    nds.bfos a0, a0, 7, 0
; CHECK-NEXT:    srai a1, a0, 31
; CHECK-NEXT:    ret
  %shl = shl i64 %a, 56
  %shr = ashr exact i64 %shl, 56
  ret i64 %shr
}

define i64 @sexti8_i64_2(i8 %a) {
; CHECK-LABEL: sexti8_i64_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    nds.bfos a0, a0, 7, 0
; CHECK-NEXT:    srai a1, a0, 31
; CHECK-NEXT:    ret
  %1 = sext i8 %a to i64
  ret i64 %1
}

define i64 @sexti16_i64(i64 %a) {
; CHECK-LABEL: sexti16_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    nds.bfos a0, a0, 15, 0
; CHECK-NEXT:    srai a1, a0, 31
; CHECK-NEXT:    ret
  %shl = shl i64 %a, 48
  %shr = ashr exact i64 %shl, 48
  ret i64 %shr
}

define i64 @sexti16_i64_2(i16 %a) {
; CHECK-LABEL: sexti16_i64_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    nds.bfos a0, a0, 15, 0
; CHECK-NEXT:    srai a1, a0, 31
; CHECK-NEXT:    ret
  %1 = sext i16 %a to i64
  ret i64 %1
}

define i64 @sexti32_i64(i64 %a) {
; CHECK-LABEL: sexti32_i64:
; CHECK:       # %bb.0:
; CHECK-NEXT:    # kill: def $x11 killed $x10
; CHECK-NEXT:    srai a1, a0, 31
; CHECK-NEXT:    ret
  %shl = shl i64 %a, 32
  %shr = ashr exact i64 %shl, 32
  ret i64 %shr
}

define i64 @sexti32_i64_2(i32 %a) {
; CHECK-LABEL: sexti32_i64_2:
; CHECK:       # %bb.0:
; CHECK-NEXT:    srai a1, a0, 31
; CHECK-NEXT:    ret
  %1 = sext i32 %a to i64
  ret i64 %1
}
