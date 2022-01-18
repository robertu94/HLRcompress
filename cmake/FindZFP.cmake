
set(ZFP_DIR "" CACHE PATH "An optional hint to a ZFP installation")

if("${ZFP_DIR}" EQUAL "")
  set(ZFP_DIR "$ENV{ZFP_DIR}")
endif()

find_path(
  ZFP_INCLUDE_DIR zfp.h
  HINTS ${ZFP_DIR} ENV ZFP_DIR
  PATH_SUFFIXES include
)

set(ZFP_PATH_SUFFIX "lib" "lib64")

find_library(
  ZFP_LIBRARY
  NAMES zfp libzfp
  HINTS ${ZFP_DIR} ENV ZFP_DIR 
  PATH_SUFFIXES ${ZFP_PATH_SUFFIX}
)

set(ZFP_LIBRARIES ${ZFP_LIBRARY})
set(ZFP_INCLUDE_DIRS ${ZFP_INCLUDE_DIR})

if("${ZFP_INCLUDE_DIR}" STREQUAL "ZFP_INCLUDE_DIR-NOTFOUND")
  set(ZFP_FOUND "0")
  message( STATUS "NOT found ZFP" )
else()
  set(ZFP_FOUND "1")
  message( STATUS "Found ZFP: ${ZFP_DIR}" )
endif()

mark_as_advanced(ZFP_LIBRARY ZFP_INCLUDE_DIR)
