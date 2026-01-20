# Exclude tests from libMLIR.so
add_mlir_library(MLIRBufferizationTestPasses
  TestOneShotModuleBufferize.cpp
  TestTensorCopyInsertion.cpp
  TestTensorLikeAndBufferLike.cpp
  TestOwnershipBasedBufferDeallocation.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRArithDialect
)
mlir_target_link_libraries(MLIRBufferizationTestPasses PUBLIC
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRSCFDialect
  MLIRIR
  MLIRPass
  MLIRTestDialect
)

target_include_directories(MLIRBufferizationTestPasses
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../Dialect/Test
  ${CMAKE_CURRENT_BINARY_DIR}/../../Dialect/Test
  )
