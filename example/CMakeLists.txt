add_executable (logmatrix logmatrix.cc)
target_link_libraries (logmatrix HLRcompress ${BLAS_LIBRARIES})

target_include_directories(logmatrix PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(logmatrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(CUDA_FOUND)
  target_include_directories(logmatrix PRIVATE ${CUDA_INCLUDE_DIRS})
  target_link_libraries(logmatrix ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY})
endif()


if(HDF5_FOUND)
  add_definitions(-DHLRCOMPRESS_USE_HDF5=1)
else()
  add_definitions(-DHLRCOMPRESS_USE_HDF5=0)
endif()

add_executable (hlrcompress hlrcompress.cc)
target_link_libraries (hlrcompress HLRcompress ${BLAS_LIBRARIES})

target_include_directories(hlrcompress PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(hlrcompress PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(TBB_FOUND)
  target_include_directories(logmatrix PRIVATE ${TBB_INCLUDE_DIRS})
  target_link_libraries(logmatrix ${TBB_LIBRARIES})

  target_include_directories(hlrcompress PRIVATE ${TBB_INCLUDE_DIRS})
  target_link_libraries(hlrcompress ${TBB_LIBRARIES})
endif()

if(ZFP_FOUND)
  target_include_directories(logmatrix PRIVATE ${ZFP_INCLUDE_DIRS})
  target_link_libraries(logmatrix ${ZFP_LIBRARIES})

  target_include_directories(hlrcompress PRIVATE ${ZFP_INCLUDE_DIRS})
  target_link_libraries(hlrcompress ${ZFP_LIBRARIES})
endif()

if(HDF5_FOUND)
  target_include_directories(hlrcompress PRIVATE ${HDF5_INCLUDE_DIRS})
  target_link_libraries (hlrcompress HLRcompress ${HDF5_LIBRARIES})
endif()

install(TARGETS hlrcompress EXPORT hlrcompress-targets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS logmatrix   EXPORT hlrcompress-targets RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
