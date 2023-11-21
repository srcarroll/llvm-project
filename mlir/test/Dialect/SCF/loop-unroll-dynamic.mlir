// RUN: mlir-opt %s -test-loop-unrolling='unroll-factor=2' -canonicalize -cse | FileCheck %s --check-prefix UNROLL-BY-2
// RUN: mlir-opt %s -test-loop-unrolling='unroll-factor=3' -canonicalize -cse | FileCheck %s --check-prefix UNROLL-BY-3


func.func @dynamic_loop_unroll(%arg0 : index, %arg1 : index, %arg2 : index,
                          %arg3: memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    memref.store %0, %arg3[%i0] : memref<?xf32>
  }
  return
}
// UNROLL-BY-2-LABEL: func @dynamic_loop_unroll
//  UNROLL-BY-2-SAME:  %[[LB:.*0]]: index,
//  UNROLL-BY-2-SAME:  %[[UB:.*1]]: index,
//  UNROLL-BY-2-SAME:  %[[STEP:.*2]]: index,
//  UNROLL-BY-2-SAME:  %[[MEM:.*3]]: memref<?xf32>
//
//   UNROLL-BY-2-DAG:  %[[V0:.*]] = arith.subi %[[UB]], %[[LB]] : index
//   UNROLL-BY-2-DAG:  %[[C1:.*]] = arith.constant 1 : index
//   UNROLL-BY-2-DAG:  %[[V1:.*]] = arith.subi %[[STEP]], %[[C1]] : index
//   UNROLL-BY-2-DAG:  %[[V2:.*]] = arith.addi %[[V0]], %[[V1]] : index
//       Compute trip count in V3.
//   UNROLL-BY-2-DAG:  %[[V3:.*]] = arith.divui %[[V2]], %[[STEP]] : index
//       Store unroll factor in C2.
//   UNROLL-BY-2-DAG:  %[[C2:.*]] = arith.constant 2 : index
//   UNROLL-BY-2-DAG:  %[[V4:.*]] = arith.remsi %[[V3]], %[[C2]] : index
//   UNROLL-BY-2-DAG:  %[[V5:.*]] = arith.subi %[[V3]], %[[V4]] : index
//   UNROLL-BY-2-DAG:  %[[V6:.*]] = arith.muli %[[V5]], %[[STEP]] : index
//       Compute upper bound of unrolled loop in V7.
//   UNROLL-BY-2-DAG:  %[[V7:.*]] = arith.addi %[[LB]], %[[V6]] : index
//       Compute step of unrolled loop in V8.
//   UNROLL-BY-2-DAG:  %[[V8:.*]] = arith.muli %[[STEP]], %[[C2]] : index
//       UNROLL-BY-2:  scf.for %[[IV:.*]] = %[[LB]] to %[[V7]] step %[[V8]] {
//  UNROLL-BY-2-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-2-NEXT:    %[[V10:.*]] = arith.addi %[[IV]], %[[STEP]] : index
//  UNROLL-BY-2-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[V10]]] : memref<?xf32>
//  UNROLL-BY-2-NEXT:  }
//  UNROLL-BY-2-NEXT:  scf.for %[[IV:.*]] = %[[V7]] to %[[UB]] step %[[STEP]] {
//  UNROLL-BY-2-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-2-NEXT:  }
//  UNROLL-BY-2-NEXT:  return

// UNROLL-BY-3-LABEL: func @dynamic_loop_unroll
//  UNROLL-BY-3-SAME:  %[[LB:.*0]]: index,
//  UNROLL-BY-3-SAME:  %[[UB:.*1]]: index,
//  UNROLL-BY-3-SAME:  %[[STEP:.*2]]: index,
//  UNROLL-BY-3-SAME:  %[[MEM:.*3]]: memref<?xf32>
//
//   UNROLL-BY-3-DAG:  %[[V0:.*]] = arith.subi %[[UB]], %[[LB]] : index
//   UNROLL-BY-3-DAG:  %[[C1:.*]] = arith.constant 1 : index
//   UNROLL-BY-3-DAG:  %[[C2:.*]] = arith.constant 2 : index
//   UNROLL-BY-3-DAG:  %[[V1:.*]] = arith.subi %[[STEP]], %[[C1]] : index
//   UNROLL-BY-3-DAG:  %[[V2:.*]] = arith.addi %[[V0]], %[[V1]] : index
//       Compute trip count in V3.
//   UNROLL-BY-3-DAG:  %[[V3:.*]] = arith.divui %[[V2]], %[[STEP]] : index
//       Store unroll factor in C3.
//   UNROLL-BY-3-DAG:  %[[C3:.*]] = arith.constant 3 : index
//   UNROLL-BY-3-DAG:  %[[V4:.*]] = arith.remsi %[[V3]], %[[C3]] : index
//   UNROLL-BY-3-DAG:  %[[V5:.*]] = arith.subi %[[V3]], %[[V4]] : index
//   UNROLL-BY-3-DAG:  %[[V6:.*]] = arith.muli %[[V5]], %[[STEP]] : index
//       Compute upper bound of unrolled loop in V7.
//   UNROLL-BY-3-DAG:  %[[V7:.*]] = arith.addi %[[LB]], %[[V6]] : index
//       Compute step of unrolled loop in V8.
//   UNROLL-BY-3-DAG:  %[[V8:.*]] = arith.muli %[[STEP]], %[[C3]] : index
//       UNROLL-BY-3:  scf.for %[[IV:.*]] = %[[LB]] to %[[V7]] step %[[V8]] {
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:    %[[V10:.*]] = arith.addi %[[IV]], %[[STEP]] : index
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[V10]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:    %[[V11:.*]] = arith.muli %[[STEP]], %[[C2]] : index
//  UNROLL-BY-3-NEXT:    %[[V12:.*]] = arith.addi %[[IV]], %[[V11]] : index
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[V12]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:  }
//  UNROLL-BY-3-NEXT:  scf.for %[[IV:.*]] = %[[V7]] to %[[UB]] step %[[STEP]] {
//  UNROLL-BY-3-NEXT:    memref.store %{{.*}}, %[[MEM]][%[[IV]]] : memref<?xf32>
//  UNROLL-BY-3-NEXT:  }
//  UNROLL-BY-3-NEXT:  return

func.func @dynamic_loop_unroll_outer_by_2(
  %arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index,
  %arg5 : index, %arg6: memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    scf.for %i1 = %arg3 to %arg4 step %arg5 {
     memref.store %0, %arg6[%i1] : memref<?xf32>
    }
  }
  return
}
// UNROLL-OUTER-BY-2-LABEL: func @dynamic_loop_unroll_outer_by_2
//  UNROLL-OUTER-BY-2-SAME:  %[[LB0:.*0]]: index,
//  UNROLL-OUTER-BY-2-SAME:  %[[UB0:.*1]]: index,
//  UNROLL-OUTER-BY-2-SAME:  %[[STEP0:.*2]]: index,
//  UNROLL-OUTER-BY-2-SAME:  %[[LB1:.*3]]: index,
//  UNROLL-OUTER-BY-2-SAME:  %[[UB1:.*4]]: index,
//  UNROLL-OUTER-BY-2-SAME:  %[[STEP1:.*5]]: index,
//  UNROLL-OUTER-BY-2-SAME:  %[[MEM:.*6]]: memref<?xf32>
//
//       UNROLL-OUTER-BY-2:  scf.for %[[IV0:.*]] = %[[LB0]] to %{{.*}} step %{{.*}} {
//  UNROLL-OUTER-BY-2-NEXT:    scf.for %[[IV1:.*]] = %[[LB1]] to %[[UB1]] step %[[STEP1]] {
//  UNROLL-OUTER-BY-2-NEXT:      memref.store %{{.*}}, %[[MEM]][%[[IV1]]] : memref<?xf32>
//  UNROLL-OUTER-BY-2-NEXT:    }
//  UNROLL-OUTER-BY-2-NEXT:    scf.for %[[IV1:.*]] = %[[LB1]] to %[[UB1]] step %[[STEP1]] {
//  UNROLL-OUTER-BY-2-NEXT:      memref.store %{{.*}}, %[[MEM]][%[[IV1]]] : memref<?xf32>
//  UNROLL-OUTER-BY-2-NEXT:    }
//  UNROLL-OUTER-BY-2-NEXT:  }
//  UNROLL-OUTER-BY-2-NEXT:  scf.for %[[IV0:.*]] = %{{.*}} to %[[UB0]] step %[[STEP0]] {
//  UNROLL-OUTER-BY-2-NEXT:    scf.for %[[IV1:.*]] = %[[LB1]] to %[[UB1]] step %[[STEP1]] {
//  UNROLL-OUTER-BY-2-NEXT:      memref.store %{{.*}}, %[[MEM]][%[[IV1]]] : memref<?xf32>
//  UNROLL-OUTER-BY-2-NEXT:    }
//  UNROLL-OUTER-BY-2-NEXT:  }
//  UNROLL-OUTER-BY-2-NEXT:  return

func.func @dynamic_loop_unroll_inner_by_2(
  %arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index, %arg4 : index,
  %arg5 : index, %arg6: memref<?xf32>) {
  %0 = arith.constant 7.0 : f32
  scf.for %i0 = %arg0 to %arg1 step %arg2 {
    scf.for %i1 = %arg3 to %arg4 step %arg5 {
     memref.store %0, %arg6[%i1] : memref<?xf32>
    }
  }
  return
}
// UNROLL-INNER-BY-2-LABEL: func @dynamic_loop_unroll_inner_by_2
//  UNROLL-INNER-BY-2-SAME:  %[[LB0:.*0]]: index,
//  UNROLL-INNER-BY-2-SAME:  %[[UB0:.*1]]: index,
//  UNROLL-INNER-BY-2-SAME:  %[[STEP0:.*2]]: index,
//  UNROLL-INNER-BY-2-SAME:  %[[LB1:.*3]]: index,
//  UNROLL-INNER-BY-2-SAME:  %[[UB1:.*4]]: index,
//  UNROLL-INNER-BY-2-SAME:  %[[STEP1:.*5]]: index,
//  UNROLL-INNER-BY-2-SAME:  %[[MEM:.*6]]: memref<?xf32>
//
//       UNROLL-INNER-BY-2:  scf.for %[[IV0:.*]] = %[[LB0]] to %[[UB0]] step %[[STEP0]] {
//       UNROLL-INNER-BY-2:    scf.for %[[IV1:.*]] = %[[LB1]] to %{{.*}} step %{{.*}} {
//  UNROLL-INNER-BY-2-NEXT:      memref.store %{{.*}}, %[[MEM]][%[[IV1]]] : memref<?xf32>
//  UNROLL-INNER-BY-2-NEXT:      %[[C1_IV:.*]] = arith.constant 1 : index
//  UNROLL-INNER-BY-2-NEXT:      %[[V0:.*]] = arith.muli %[[STEP1]], %[[C1_IV]] : index
//  UNROLL-INNER-BY-2-NEXT:      %[[V1:.*]] = arith.addi %[[IV1]], %[[V0]] : index
//  UNROLL-INNER-BY-2-NEXT:      memref.store %{{.*}}, %[[MEM]][%[[V1]]] : memref<?xf32>
//  UNROLL-INNER-BY-2-NEXT:    }
//  UNROLL-INNER-BY-2-NEXT:    scf.for %[[IV1:.*]] = %{{.*}} to %[[UB1]] step %[[STEP1]] {
//  UNROLL-INNER-BY-2-NEXT:      memref.store %{{.*}}, %[[MEM]][%[[IV1]]] : memref<?xf32>
//  UNROLL-INNER-BY-2-NEXT:    }
//  UNROLL-INNER-BY-2-NEXT:  }
//  UNROLL-INNER-BY-2-NEXT:  return
