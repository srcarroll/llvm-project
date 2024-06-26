; RUN: opt %s -dxil-embed -S -o - | FileCheck %s
; RUN: llc %s --filetype=obj -o - | obj2yaml | FileCheck %s --check-prefix=DXC
target triple = "dxil-unknown-shadermodel6.5-library"

; Make sure triple is restored after updated to dxil.
; CHECK:target triple = "dxil-unknown-shadermodel6.5-library"

define i32 @add(i32 %a, i32 %b) {
  %sum = add i32 %a, %b
  ret i32 %sum
}

; CHECK: @dx.dxil = private constant [[BC_TYPE:\[[0-9]+ x i8\]]] c"BC\C0\DE{{[^"]+}}", section "DXIL", align 4

; The dxil global should be the first here because we generate it before the
; other globals. If it isn't the first here, that's probably a bug.
; CHECK: @llvm.compiler.used = appending global {{\[[0-9]+ x ptr\]}} [ptr @dx.dxil

; This is using regex matches on some sizes, offsets and fields. These are all
; going to change as the DirectX backend continues to evolve and implement more
; features. Rather than extending this test to cover those future features, this
; test's matches are extremely fuzzy so that it won't break.

; DXC: --- !dxcontainer
; DXC-NEXT: Header:
; DXC-NEXT:   Hash:            [ 0x0, 0x0, 0x0,
; DXC:   Version:
; DXC-NEXT:     Major:           1
; DXC-NEXT:     Minor:           0
; DXC-NEXT:   FileSize:        [[#]]
; DXC-NEXT:   PartCount:       [[#]]
; DXC-NEXT:   PartOffsets:     [ {{[0-9, ]+}} ]
; DXC-NEXT: Parts:

; In verifying the DXIL part, this test captures the size of the part, and
; derives the program header and dxil size fields from the part's size.

; DXC:   - Name:            DXIL
; DXC-NEXT:     Size:            [[#SIZE:]]
; DXC-NEXT:     Program:
; DXC-NEXT:       MajorVersion:    6
; DXC-NEXT:       MinorVersion:    5
; DXC-NEXT:       ShaderKind:      6
; DXC-NEXT:       Size:            [[#div(SIZE,4)]]
; DXC-NEXT:       DXILMajorVersion: 1
; DXC-NEXT:       DXILMinorVersion: 5
; DXC-NEXT:       DXILSize:        [[#SIZE - 24]]
; DXC-NEXT:       DXIL:            [ 0x42, 0x43, 0xC0, 0xDE,
; DXC:      - Name:            SFI0
; DXC-NEXT:   Size:            8
; DXC-NOT:    Flags:
