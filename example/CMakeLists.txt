add_executable (compress compress.cc)
target_link_libraries (compress HLRcompress blas lapack)
target_include_directories(compress PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
