# 0. clear var
UNSET (_DEV_OCL_HEADER_PATH)
UNSET (_OCL_BASE_SOURCE)
UNSET (_OCL_OPS_SOURCE)
UNSET (_DEV_OCL_DEVICE_SOURCE)
UNSET (_DEV_OCL_COMPILER_DEFINES)
UNSET (_DEV_OCL_COMPILER_OPTIONS)
UNSET (_DEV_OCL_LINKER_OPTIONS)
UNSET (_DEV_OCL_LINK_LIBRARIES)


# 1.  set source root path
SET(_OCL_ROOT ${CMAKE_SOURCE_DIR}/source/device/opencl)


# 2.  add header file path
LIST (APPEND _DEV_OCL_HEADER_PATH           ${_OCL_ROOT})
LIST (APPEND _DEV_OCL_HEADER_PATH           ${_OCL_ROOT}/include)

# 4.  add source files
AUX_SOURCE_DIRECTORY("${_OCL_ROOT}"         _OCL_BASE_SOURCE)
AUX_SOURCE_DIRECTORY("${_OCL_ROOT}/oppack4"  _OCL_OPS_SOURCE_PACK4)
LIST (APPEND _DEV_OCL_DEVICE_SOURCE         ${_OCL_BASE_SOURCE})
LIST (APPEND _DEV_OCL_DEVICE_SOURCE         ${_OCL_OPS_SOURCE})
LIST (APPEND _DEV_OCL_DEVICE_SOURCE         ${_OCL_OPS_SOURCE_PACK4})


# 5.  add build options for cpu device
# 5.1 is a gcc or clang like compiler
IF (TENGINE_COMPILER_GCC OR TENGINE_COMPILER_CLANG)
    IF (TENGINE_COMPILER_GCC AND (${CMAKE_CXX_COMPILER_VERSION} VERSION_GREATER_EQUAL "6.1"))
        LIST (APPEND _DEV_OCL_COMPILER_OPTIONS -Wno-ignored-attributes)
    ENDIF()
ENDIF()

# 5.2 is Microsoft Visual C++
IF (TENGINE_COMPILER_MSVC)
ENDIF()


# 6.  add link options
OPTION(TENGINE_OPENCL_PROFILE_TIME "enable opencl profile time" OFF)
IF (TENGINE_OPENCL_PROFILE_TIME)
    LIST (APPEND _DEV_OCL_COMPILER_DEFINES OPENCL_PROFILE_TIME)
ENDIF ()

# 7.  add link libs
FIND_PACKAGE(OpenCL)
IF(NOT OpenCL_FOUND)
    message(WARNING "please set cmake -DOpenCL_LIBRARY=user/lib/... manual")
ENDIF()
LIST (APPEND _DEV_OCL_LINK_LIBRARIES  ${OpenCL_LIBRARY})


# 8. set all to cmake cache
SET (TENGINE_OCL_HEADER_PATH       ${_DEV_OCL_HEADER_PATH}        CACHE INTERNAL  "Tengine OpenCL device header files searching path"   FORCE)
SET (TENGINE_OCL_LINK_PATH         ${_DEV_OCL_LINK_PATH}          CACHE INTERNAL  "Tengine OpenCL device link libraries searching path" FORCE)
SET (TENGINE_OCL_DEVICE_SOURCE     ${_DEV_OCL_DEVICE_SOURCE}      CACHE INTERNAL  "Tengine OpenCL device main source files"             FORCE)
SET (TENGINE_OCL_COMPILER_DEFINES  ${_DEV_OCL_COMPILER_DEFINES}   CACHE INTERNAL  "Tengine OpenCL about compiler defines"               FORCE)
SET (TENGINE_OCL_COMPILER_OPTIONS  ${_DEV_OCL_COMPILER_OPTIONS}   CACHE INTERNAL  "Tengine OpenCL about compiler options"               FORCE)
SET (TENGINE_OCL_LINKER_OPTIONS    ${_DEV_OCL_LINKER_OPTIONS}     CACHE INTERNAL  "Tengine OpenCL about linker options"                 FORCE)
SET (TENGINE_OCL_LINK_LIBRARIES    ${_DEV_OCL_LINK_LIBRARIES}     CACHE INTERNAL  "Tengine OpenCL about link libraries"                 FORCE)


# 9. install device option
INSTALL (FILES ${_OCL_ROOT}/ocl_define.h DESTINATION include/tengine RENAME ocl_device.h)
