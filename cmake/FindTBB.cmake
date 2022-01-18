
set(TBB_DIR "" CACHE PATH "An optional hint to a TBB installation")

if("${TBB_DIR}" EQUAL "")
  set(TBB_DIR "$ENV{TBB_DIR}")
endif()

find_package(PkgConfig QUIET)
pkg_check_modules(PC_TBB QUIET tbb)

find_path(
  TBB_INCLUDE_DIR tbb/tbb.h
  HINTS ${TBB_DIR} ENV TBB_DIR ${PC_TBB_INCLUDEDIR} ${PC_TBB_INCLUDE_DIRS}
  PATH_SUFFIXES include
)

set(TBB_PATH_SUFFIX "lib" "lib64" "lib/intel64" "lib/intel64/gcc4.4")

find_library(
  TBB_LIBRARY
  NAMES tbb libtbb
  HINTS ${TBB_DIR} ENV TBB_DIR ${PC_TBB_LIBDIR} ${PC_TBB_LIBRARY_DIRS}
  PATH_SUFFIXES ${TBB_PATH_SUFFIX}
)

set(TBB_LIBRARIES ${TBB_LIBRARY})
set(TBB_INCLUDE_DIRS ${TBB_INCLUDE_DIR})

if("${TBB_INCLUDE_DIR}" STREQUAL "TBB_INCLUDE_DIR-NOTFOUND")
  set(TBB_FOUND "0")
  message( STATUS "NOT found TBB" )
else()
  set(TBB_FOUND "1")
  message( STATUS "Found TBB: ${TBB_DIR}" )
endif()

mark_as_advanced(TBB_LIBRARY TBB_INCLUDE_DIR)
