# NOTE: Assertions have been autogenerated by utils/update_mca_test_checks.py
# RUN: llvm-mca -mtriple=x86_64-unknown-unknown -mcpu=alderlake -instruction-tables < %s | FileCheck %s

pabsb       %mm0, %mm2
pabsb       (%rax), %mm2

pabsb       %xmm0, %xmm2
pabsb       (%rax), %xmm2

pabsd       %mm0, %mm2
pabsd       (%rax), %mm2

pabsd       %xmm0, %xmm2
pabsd       (%rax), %xmm2

pabsw       %mm0, %mm2
pabsw       (%rax), %mm2

pabsw       %xmm0, %xmm2
pabsw       (%rax), %xmm2

palignr     $1, %mm0, %mm2
palignr     $1, (%rax), %mm2

palignr     $1, %xmm0, %xmm2
palignr     $1, (%rax), %xmm2

phaddd      %mm0, %mm2
phaddd      (%rax), %mm2

phaddd      %xmm0, %xmm2
phaddd      (%rax), %xmm2

phaddsw     %mm0, %mm2
phaddsw     (%rax), %mm2

phaddsw     %xmm0, %xmm2
phaddsw     (%rax), %xmm2

phaddw      %mm0, %mm2
phaddw      (%rax), %mm2

phaddw      %xmm0, %xmm2
phaddw      (%rax), %xmm2

phsubd      %mm0, %mm2
phsubd      (%rax), %mm2

phsubd      %xmm0, %xmm2
phsubd      (%rax), %xmm2

phsubsw     %mm0, %mm2
phsubsw     (%rax), %mm2

phsubsw     %xmm0, %xmm2
phsubsw     (%rax), %xmm2

phsubw      %mm0, %mm2
phsubw      (%rax), %mm2

phsubw      %xmm0, %xmm2
phsubw      (%rax), %xmm2

pmaddubsw   %mm0, %mm2
pmaddubsw   (%rax), %mm2

pmaddubsw   %xmm0, %xmm2
pmaddubsw   (%rax), %xmm2

pmulhrsw    %mm0, %mm2
pmulhrsw    (%rax), %mm2

pmulhrsw    %xmm0, %xmm2
pmulhrsw    (%rax), %xmm2

pshufb      %mm0, %mm2
pshufb      (%rax), %mm2

pshufb      %xmm0, %xmm2
pshufb      (%rax), %xmm2

psignb      %mm0, %mm2
psignb      (%rax), %mm2

psignb      %xmm0, %xmm2
psignb      (%rax), %xmm2

psignd      %mm0, %mm2
psignd      (%rax), %mm2

psignd      %xmm0, %xmm2
psignd      (%rax), %xmm2

psignw      %mm0, %mm2
psignw      (%rax), %mm2

psignw      %xmm0, %xmm2
psignw      (%rax), %xmm2

# CHECK:      Instruction Info:
# CHECK-NEXT: [1]: #uOps
# CHECK-NEXT: [2]: Latency
# CHECK-NEXT: [3]: RThroughput
# CHECK-NEXT: [4]: MayLoad
# CHECK-NEXT: [5]: MayStore
# CHECK-NEXT: [6]: HasSideEffects (U)

# CHECK:      [1]    [2]    [3]    [4]    [5]    [6]    Instructions:
# CHECK-NEXT:  1      1     1.00                        pabsb	%mm0, %mm2
# CHECK-NEXT:  2      9     1.00    *                   pabsb	(%rax), %mm2
# CHECK-NEXT:  1      1     0.50                        pabsb	%xmm0, %xmm2
# CHECK-NEXT:  2      8     0.50    *                   pabsb	(%rax), %xmm2
# CHECK-NEXT:  1      1     1.00                        pabsd	%mm0, %mm2
# CHECK-NEXT:  2      9     1.00    *                   pabsd	(%rax), %mm2
# CHECK-NEXT:  1      1     0.50                        pabsd	%xmm0, %xmm2
# CHECK-NEXT:  2      8     0.50    *                   pabsd	(%rax), %xmm2
# CHECK-NEXT:  1      1     1.00                        pabsw	%mm0, %mm2
# CHECK-NEXT:  2      9     1.00    *                   pabsw	(%rax), %mm2
# CHECK-NEXT:  1      1     0.50                        pabsw	%xmm0, %xmm2
# CHECK-NEXT:  2      8     0.50    *                   pabsw	(%rax), %xmm2
# CHECK-NEXT:  1      1     1.00                        palignr	$1, %mm0, %mm2
# CHECK-NEXT:  2      9     1.00    *                   palignr	$1, (%rax), %mm2
# CHECK-NEXT:  1      1     1.00                        palignr	$1, %xmm0, %xmm2
# CHECK-NEXT:  2      8     1.00    *                   palignr	$1, (%rax), %xmm2
# CHECK-NEXT:  3      3     2.00                        phaddd	%mm0, %mm2
# CHECK-NEXT:  4      11    2.00    *                   phaddd	(%rax), %mm2
# CHECK-NEXT:  3      2     1.00                        phaddd	%xmm0, %xmm2
# CHECK-NEXT:  4      9     1.00    *                   phaddd	(%rax), %xmm2
# CHECK-NEXT:  3      3     2.00                        phaddsw	%mm0, %mm2
# CHECK-NEXT:  4      11    2.00    *                   phaddsw	(%rax), %mm2
# CHECK-NEXT:  3      2     1.00                        phaddsw	%xmm0, %xmm2
# CHECK-NEXT:  4      9     1.00    *                   phaddsw	(%rax), %xmm2
# CHECK-NEXT:  3      3     2.00                        phaddw	%mm0, %mm2
# CHECK-NEXT:  4      11    2.00    *                   phaddw	(%rax), %mm2
# CHECK-NEXT:  3      2     1.00                        phaddw	%xmm0, %xmm2
# CHECK-NEXT:  4      9     1.00    *                   phaddw	(%rax), %xmm2
# CHECK-NEXT:  3      3     2.00                        phsubd	%mm0, %mm2
# CHECK-NEXT:  4      11    2.00    *                   phsubd	(%rax), %mm2
# CHECK-NEXT:  3      2     1.00                        phsubd	%xmm0, %xmm2
# CHECK-NEXT:  4      9     1.00    *                   phsubd	(%rax), %xmm2
# CHECK-NEXT:  3      3     2.00                        phsubsw	%mm0, %mm2
# CHECK-NEXT:  4      11    2.00    *                   phsubsw	(%rax), %mm2
# CHECK-NEXT:  3      2     1.00                        phsubsw	%xmm0, %xmm2
# CHECK-NEXT:  4      9     1.00    *                   phsubsw	(%rax), %xmm2
# CHECK-NEXT:  3      3     2.00                        phsubw	%mm0, %mm2
# CHECK-NEXT:  4      11    2.00    *                   phsubw	(%rax), %mm2
# CHECK-NEXT:  3      2     1.00                        phsubw	%xmm0, %xmm2
# CHECK-NEXT:  4      9     1.00    *                   phsubw	(%rax), %xmm2
# CHECK-NEXT:  1      5     1.00                        pmaddubsw	%mm0, %mm2
# CHECK-NEXT:  2      13    1.00    *                   pmaddubsw	(%rax), %mm2
# CHECK-NEXT:  1      5     0.50                        pmaddubsw	%xmm0, %xmm2
# CHECK-NEXT:  2      13    0.50    *                   pmaddubsw	(%rax), %xmm2
# CHECK-NEXT:  1      5     1.00                        pmulhrsw	%mm0, %mm2
# CHECK-NEXT:  2      13    1.00    *                   pmulhrsw	(%rax), %mm2
# CHECK-NEXT:  1      5     0.50                        pmulhrsw	%xmm0, %xmm2
# CHECK-NEXT:  2      13    0.50    *                   pmulhrsw	(%rax), %xmm2
# CHECK-NEXT:  2      3     1.00                        pshufb	%mm0, %mm2
# CHECK-NEXT:  3      11    1.00    *                   pshufb	(%rax), %mm2
# CHECK-NEXT:  1      1     0.50                        pshufb	%xmm0, %xmm2
# CHECK-NEXT:  2      8     0.50    *                   pshufb	(%rax), %xmm2
# CHECK-NEXT:  1      1     1.00                        psignb	%mm0, %mm2
# CHECK-NEXT:  2      9     1.00    *                   psignb	(%rax), %mm2
# CHECK-NEXT:  1      1     0.50                        psignb	%xmm0, %xmm2
# CHECK-NEXT:  2      8     0.50    *                   psignb	(%rax), %xmm2
# CHECK-NEXT:  1      1     1.00                        psignd	%mm0, %mm2
# CHECK-NEXT:  2      9     1.00    *                   psignd	(%rax), %mm2
# CHECK-NEXT:  1      1     0.50                        psignd	%xmm0, %xmm2
# CHECK-NEXT:  2      8     0.50    *                   psignd	(%rax), %xmm2
# CHECK-NEXT:  1      1     1.00                        psignw	%mm0, %mm2
# CHECK-NEXT:  2      9     1.00    *                   psignw	(%rax), %mm2
# CHECK-NEXT:  1      1     0.50                        psignw	%xmm0, %xmm2
# CHECK-NEXT:  2      8     0.50    *                   psignw	(%rax), %xmm2

# CHECK:      Resources:
# CHECK-NEXT: [0]   - ADLPPort00
# CHECK-NEXT: [1]   - ADLPPort01
# CHECK-NEXT: [2]   - ADLPPort02
# CHECK-NEXT: [3]   - ADLPPort03
# CHECK-NEXT: [4]   - ADLPPort04
# CHECK-NEXT: [5]   - ADLPPort05
# CHECK-NEXT: [6]   - ADLPPort06
# CHECK-NEXT: [7]   - ADLPPort07
# CHECK-NEXT: [8]   - ADLPPort08
# CHECK-NEXT: [9]   - ADLPPort09
# CHECK-NEXT: [10]  - ADLPPort10
# CHECK-NEXT: [11]  - ADLPPort11
# CHECK-NEXT: [12]  - ADLPPortInvalid

# CHECK:      Resource pressure per iteration:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]
# CHECK-NEXT: 38.67  25.67  10.67  10.67   -     49.67   -      -      -      -     10.67   -      -

# CHECK:      Resource pressure by instruction:
# CHECK-NEXT: [0]    [1]    [2]    [3]    [4]    [5]    [6]    [7]    [8]    [9]    [10]   [11]   [12]   Instructions:
# CHECK-NEXT: 1.00    -      -      -      -      -      -      -      -      -      -      -      -     pabsb	%mm0, %mm2
# CHECK-NEXT: 1.00    -     0.33   0.33    -      -      -      -      -      -     0.33    -      -     pabsb	(%rax), %mm2
# CHECK-NEXT: 0.50   0.50    -      -      -      -      -      -      -      -      -      -      -     pabsb	%xmm0, %xmm2
# CHECK-NEXT: 0.50   0.50   0.33   0.33    -      -      -      -      -      -     0.33    -      -     pabsb	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -      -      -      -      -      -      -      -      -      -      -     pabsd	%mm0, %mm2
# CHECK-NEXT: 1.00    -     0.33   0.33    -      -      -      -      -      -     0.33    -      -     pabsd	(%rax), %mm2
# CHECK-NEXT: 0.50   0.50    -      -      -      -      -      -      -      -      -      -      -     pabsd	%xmm0, %xmm2
# CHECK-NEXT: 0.50   0.50   0.33   0.33    -      -      -      -      -      -     0.33    -      -     pabsd	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -      -      -      -      -      -      -      -      -      -      -     pabsw	%mm0, %mm2
# CHECK-NEXT: 1.00    -     0.33   0.33    -      -      -      -      -      -     0.33    -      -     pabsw	(%rax), %mm2
# CHECK-NEXT: 0.50   0.50    -      -      -      -      -      -      -      -      -      -      -     pabsw	%xmm0, %xmm2
# CHECK-NEXT: 0.50   0.50   0.33   0.33    -      -      -      -      -      -     0.33    -      -     pabsw	(%rax), %xmm2
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -      -      -     palignr	$1, %mm0, %mm2
# CHECK-NEXT:  -      -     0.33   0.33    -     1.00    -      -      -      -     0.33    -      -     palignr	$1, (%rax), %mm2
# CHECK-NEXT:  -      -      -      -      -     1.00    -      -      -      -      -      -      -     palignr	$1, %xmm0, %xmm2
# CHECK-NEXT:  -      -     0.33   0.33    -     1.00    -      -      -      -     0.33    -      -     palignr	$1, (%rax), %xmm2
# CHECK-NEXT: 0.50    -      -      -      -     2.50    -      -      -      -      -      -      -     phaddd	%mm0, %mm2
# CHECK-NEXT: 0.50    -     0.33   0.33    -     2.50    -      -      -      -     0.33    -      -     phaddd	(%rax), %mm2
# CHECK-NEXT: 0.33   1.33    -      -      -     1.33    -      -      -      -      -      -      -     phaddd	%xmm0, %xmm2
# CHECK-NEXT: 0.33   1.33   0.33   0.33    -     1.33    -      -      -      -     0.33    -      -     phaddd	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -      -      -     2.00    -      -      -      -      -      -      -     phaddsw	%mm0, %mm2
# CHECK-NEXT: 1.00    -     0.33   0.33    -     2.00    -      -      -      -     0.33    -      -     phaddsw	(%rax), %mm2
# CHECK-NEXT: 0.50   1.50    -      -      -     1.00    -      -      -      -      -      -      -     phaddsw	%xmm0, %xmm2
# CHECK-NEXT: 0.50   1.50   0.33   0.33    -     1.00    -      -      -      -     0.33    -      -     phaddsw	(%rax), %xmm2
# CHECK-NEXT: 0.50    -      -      -      -     2.50    -      -      -      -      -      -      -     phaddw	%mm0, %mm2
# CHECK-NEXT: 0.50    -     0.33   0.33    -     2.50    -      -      -      -     0.33    -      -     phaddw	(%rax), %mm2
# CHECK-NEXT: 0.33   1.33    -      -      -     1.33    -      -      -      -      -      -      -     phaddw	%xmm0, %xmm2
# CHECK-NEXT: 0.33   1.33   0.33   0.33    -     1.33    -      -      -      -     0.33    -      -     phaddw	(%rax), %xmm2
# CHECK-NEXT: 0.50    -      -      -      -     2.50    -      -      -      -      -      -      -     phsubd	%mm0, %mm2
# CHECK-NEXT: 0.50    -     0.33   0.33    -     2.50    -      -      -      -     0.33    -      -     phsubd	(%rax), %mm2
# CHECK-NEXT: 0.33   1.33    -      -      -     1.33    -      -      -      -      -      -      -     phsubd	%xmm0, %xmm2
# CHECK-NEXT: 0.33   1.33   0.33   0.33    -     1.33    -      -      -      -     0.33    -      -     phsubd	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -      -      -     2.00    -      -      -      -      -      -      -     phsubsw	%mm0, %mm2
# CHECK-NEXT: 1.00    -     0.33   0.33    -     2.00    -      -      -      -     0.33    -      -     phsubsw	(%rax), %mm2
# CHECK-NEXT: 0.50   1.50    -      -      -     1.00    -      -      -      -      -      -      -     phsubsw	%xmm0, %xmm2
# CHECK-NEXT: 0.50   1.50   0.33   0.33    -     1.00    -      -      -      -     0.33    -      -     phsubsw	(%rax), %xmm2
# CHECK-NEXT: 0.50    -      -      -      -     2.50    -      -      -      -      -      -      -     phsubw	%mm0, %mm2
# CHECK-NEXT: 0.50    -     0.33   0.33    -     2.50    -      -      -      -     0.33    -      -     phsubw	(%rax), %mm2
# CHECK-NEXT: 0.33   1.33    -      -      -     1.33    -      -      -      -      -      -      -     phsubw	%xmm0, %xmm2
# CHECK-NEXT: 0.33   1.33   0.33   0.33    -     1.33    -      -      -      -     0.33    -      -     phsubw	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -      -      -      -      -      -      -      -      -      -      -     pmaddubsw	%mm0, %mm2
# CHECK-NEXT: 1.00    -     0.33   0.33    -      -      -      -      -      -     0.33    -      -     pmaddubsw	(%rax), %mm2
# CHECK-NEXT: 0.50   0.50    -      -      -      -      -      -      -      -      -      -      -     pmaddubsw	%xmm0, %xmm2
# CHECK-NEXT: 0.50   0.50   0.33   0.33    -      -      -      -      -      -     0.33    -      -     pmaddubsw	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -      -      -      -      -      -      -      -      -      -      -     pmulhrsw	%mm0, %mm2
# CHECK-NEXT: 1.00    -     0.33   0.33    -      -      -      -      -      -     0.33    -      -     pmulhrsw	(%rax), %mm2
# CHECK-NEXT: 0.50   0.50    -      -      -      -      -      -      -      -      -      -      -     pmulhrsw	%xmm0, %xmm2
# CHECK-NEXT: 0.50   0.50   0.33   0.33    -      -      -      -      -      -     0.33    -      -     pmulhrsw	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -      -      -     1.00    -      -      -      -      -      -      -     pshufb	%mm0, %mm2
# CHECK-NEXT: 1.00    -     0.33   0.33    -     1.00    -      -      -      -     0.33    -      -     pshufb	(%rax), %mm2
# CHECK-NEXT:  -     0.50    -      -      -     0.50    -      -      -      -      -      -      -     pshufb	%xmm0, %xmm2
# CHECK-NEXT:  -     0.50   0.33   0.33    -     0.50    -      -      -      -     0.33    -      -     pshufb	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -      -      -      -      -      -      -      -      -      -      -     psignb	%mm0, %mm2
# CHECK-NEXT: 1.00    -     0.33   0.33    -      -      -      -      -      -     0.33    -      -     psignb	(%rax), %mm2
# CHECK-NEXT: 0.50   0.50    -      -      -      -      -      -      -      -      -      -      -     psignb	%xmm0, %xmm2
# CHECK-NEXT: 0.50   0.50   0.33   0.33    -      -      -      -      -      -     0.33    -      -     psignb	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -      -      -      -      -      -      -      -      -      -      -     psignd	%mm0, %mm2
# CHECK-NEXT: 1.00    -     0.33   0.33    -      -      -      -      -      -     0.33    -      -     psignd	(%rax), %mm2
# CHECK-NEXT: 0.50   0.50    -      -      -      -      -      -      -      -      -      -      -     psignd	%xmm0, %xmm2
# CHECK-NEXT: 0.50   0.50   0.33   0.33    -      -      -      -      -      -     0.33    -      -     psignd	(%rax), %xmm2
# CHECK-NEXT: 1.00    -      -      -      -      -      -      -      -      -      -      -      -     psignw	%mm0, %mm2
# CHECK-NEXT: 1.00    -     0.33   0.33    -      -      -      -      -      -     0.33    -      -     psignw	(%rax), %mm2
# CHECK-NEXT: 0.50   0.50    -      -      -      -      -      -      -      -      -      -      -     psignw	%xmm0, %xmm2
# CHECK-NEXT: 0.50   0.50   0.33   0.33    -      -      -      -      -      -     0.33    -      -     psignw	(%rax), %xmm2
