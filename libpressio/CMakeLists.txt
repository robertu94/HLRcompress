add_library(libpressio_hlrcompress
  libpressio_hlrcompress.cc
  )
target_link_libraries(
  libpressio_hlrcompress
  PRIVATE
  LibPressio::libpressio
  HLRcompress
  )

if(CUDA_FOUND)
  target_link_libraries(
    libpressio_hlrcompress
    PRIVATE
    ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} ${CUDA_cusolver_LIBRARY}
    )
endif()
target_link_libraries(
  libpressio_hlrcompress
  PRIVATE
  ${BLAS_LIBRARIES}
  )


target_link_libraries(
  libpressio_hlrcompress
  PRIVATE
  ${TBB_LIBRARIES} 
  )

target_link_libraries(
  libpressio_hlrcompress
  PRIVATE
  ${HDF5_LIBRARIES})


install(TARGETS libpressio_hlrcompress
  EXPORT LibPressioHLRCompressTargets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT LibPressioHLRCompressTargets NAMESPACE LibPressioHLRCompress:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibPressioHLRCompress)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/LibPressioHLRCompressConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/LibPressioHLRCompressConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibPressioHLRCompress
  )
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/LibPressioHLRCompressConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
  )
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/LibPressioHLRCompressConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/LibPressioHLRCompressConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibPressioHLRCompress
  )
