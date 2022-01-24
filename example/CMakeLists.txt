add_executable (logmatrix logmatrix.cc)
target_link_libraries (logmatrix HLRcompress ${BLAS_LIBRARIES})

if(TBB_FOUND)
  target_include_directories(logmatrix PRIVATE ${TBB_INCLUDE_DIRS})
  target_link_libraries(logmatrix ${TBB_LIBRARIES})
endif()

if(ZFP_FOUND)
  target_include_directories(logmatrix PRIVATE ${ZFP_INCLUDE_DIRS})
  target_link_libraries(logmatrix ${ZFP_LIBRARIES})
endif()

target_include_directories(logmatrix PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(logmatrix PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)


if(HDF5_FOUND)
  
add_executable (h5compress h5compress.cc)
target_include_directories(h5compress PRIVATE ${HDF5_CXX_INCLUDE_DIRS})
target_link_libraries (h5compress HLRcompress ${BLAS_LIBRARIES} ${HDF5_CXX_LIBRARIES})

if(TBB_FOUND)
  target_include_directories(h5compress PRIVATE ${TBB_INCLUDE_DIRS})
  target_link_libraries(h5compress ${TBB_LIBRARIES})
endif()

if(ZFP_FOUND)
  target_include_directories(h5compress PRIVATE ${ZFP_INCLUDE_DIRS})
  target_link_libraries(h5compress ${ZFP_LIBRARIES})
endif()

target_include_directories(h5compress PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(h5compress PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)

endif()
