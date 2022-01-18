add_executable (compress compress.cc)
target_link_libraries (compress HLRcompress blas lapack)

if(TBB_FOUND)
  target_include_directories(compress PRIVATE ${TBB_INCLUDE_DIRS})
  target_link_libraries(compress ${TBB_LIBRARIES})
endif()

if(ZFP_FOUND)
  target_include_directories(compress PRIVATE ${ZFP_INCLUDE_DIRS})
  target_link_libraries(compress ${ZFP_LIBRARIES})
endif()

target_include_directories(compress PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(compress PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
