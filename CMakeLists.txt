cmake_minimum_required (VERSION 3.10.2)

# disable in-source builds
set (CMAKE_DISABLE_SOURCE_CHANGES ON)
set (CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# define the project
project (HLRcompress
  VERSION 0.1
  DESCRIPTION "Hierarchical Low-Rank Compression"
  LANGUAGES CXX
  )
enable_testing()

# guard against bad build-type strings
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type was set. Setting build type to ${default_build_type}.")
  set(CMAKE_BUILD_TYPE ${default_build_type} CACHE
    STRING "Choose the type to build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# set the Compiler ID for clang on macOS to AppleClang
if (POLICY CMP0025)
  cmake_policy (SET CMP0025 NEW)
endif()

# compile with C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

# option(BUILD_SHARED_LIBS "build shared libraries over static libraries" ON)

# header only so no testing
option(ENABLE_TESTING OFF)

#find dependencies
include(GNUInstallDirs)
find_package(PkgConfig)
pkg_search_module(TBB  IMPORTED_TARGET libtbb)

find_package(OpenMP)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

add_subdirectory(hlrcompress)
add_subdirectory(example)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/hlrcompress/config.h.in
  ${CMAKE_BINARY_DIR}/hlrcompress/config.h
  @ONLY
)
