cmake_minimum_required (VERSION 3.12)

# disable in-source builds
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# define the project
project(HLRcompress
  VERSION 0.1
  DESCRIPTION "Hierarchical Low-Rank Compression"
  LANGUAGES C CXX
  )
enable_testing()

# guard against bad build-type strings
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type was set. Setting build type to ${default_build_type}.")
  set(CMAKE_BUILD_TYPE ${default_build_type} CACHE
    STRING "Choose the type to build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# compile with standard C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# option(BUILD_SHARED_LIBS "build shared libraries over static libraries" ON)

# header only so no testing
option(ENABLE_TESTING OFF)

#find dependencies
include(GNUInstallDirs)

find_package(TBB)

find_package(OpenMP)
if(OPENMP_FOUND)
  set(OPENMP_FOUND "1")
else()
  set(OPENMP_FOUND "0")
endif()

option(HLRCOMPRESS_USE_ILP64 "use ILP64 interface to BLAS/LAPACK" OFF)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

message( STATUS "BLAS:   ${BLAS_LIBRARIES}")

if(HLRCOMPRESS_USE_ILP64)
  set(HLRCOMPRESS_USE_ILP64 "1")
else()
  set(HLRCOMPRESS_USE_ILP64 "0")
endif()

find_package(ZFP)
find_package(HDF5 COMPONENTS C)
find_package(CUDA)

message( STATUS "cuSolver: ${CUDA_cusolver_LIBRARY}")

if(CUDA_FOUND)
  set(CUDA_FOUND "1")
else()
  set(CUDA_FOUND "0")
endif()

add_subdirectory(hlrcompress)
add_subdirectory(example)

set(HLRCOMPRESS_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/hlrcompress")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/hlrcompress/config.h.in
  ${CMAKE_BINARY_DIR}/hlrcompress/config.h
  @ONLY
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hlrcompress/config.h DESTINATION ${HLRCOMPRESS_INSTALL_INCLUDEDIR})

