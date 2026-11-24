
# Prevent compilation in-source
IF( ${CMAKE_BINARY_DIR} STREQUAL ${PROJECT_SOURCE_DIR} )
  MESSAGE( " " )
  MESSAGE( FATAL_ERROR "Source and build  directories are the same.
 Create an empty build directory,
 change into it and re-invoke cmake")
ENDIF()

#    Tell CMake where to leave executables
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR})
IF(APPLE)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/macos/)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/macos/)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/macos/)
ELSEIF(UNIX)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/linux/)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/linux/)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/linux/)
ELSEIF(WIN32)
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/win32/)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib/win32/)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/win32/)
ENDIF()

# Build Type. By default is release
# SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug, Release" FORCE)

#Path of header files relative to this file path
SET(HEADER_PATH ${PROJECT_SOURCE_DIR}/header/)
MESSAGE(${HEADER_PATH})
include_directories( ${PROJECT_SOURCE_DIR} )


# ### Search Additional Libraries ##########
FIND_PACKAGE(OpenCV REQUIRED)
IF (NOT APPLE)
  FIND_PACKAGE(OpenMP)
ENDIF ()
# #### End of additional libraries search ##########


IF (OpenCV_FOUND)
  MESSAGE("OpenCV Found")
  include_directories(${OpenCV_INCLUDE_DIRS})
  link_directories(${OpenCV_LIB_DIRS})
  SET( SYSTEM_LIBS  ${SYSTEM_LIBS}  ${OpenCV_LIBS} )
ENDIF()

IF (OPENMP_FOUND)
  MESSAGE("OpenMP Found")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
ENDIF(OPENMP_FOUND)

FIND_PACKAGE(CUDA QUIET)
IF(NOT CUDA_FOUND)
  MESSAGE("No CUDA Found")
  SET(NO_CUDA_BUILD 1)
ENDIF()

IF (NOT NO_CUDA_BUILD)
  FIND_PACKAGE(CUDA REQUIRED)
  MESSAGE("CUDA Found")

  SET (CUDA_SAMPLES_DIR "/usr/local/cuda/samples" CACHE PATH "Path to CUDA Samples Directory")
  SET (NVTX_INCLUDE_DIR "/usr/local/cuda/include" CACHE PATH "Path to NVTX include file")
  SET (NVTX_LIB_DIR "/usr/local/cuda/lib64" CACHE PATH "Path to NVTX lib file")
  include_directories(${CUDA_SAMPLES_DIR}/common/inc)
  include_directories(${NVTX_INCLUDE_DIR})
  link_directories(${NVTX_LIB_DIR})

  SET (NPP_CUDA
    ${CUDA_nppial_LIBRARY}
    ${CUDA_nppicc_LIBRARY}
    ${CUDA_nppidei_LIBRARY}
    ${CUDA_nppif_LIBRARY}
    ${CUDA_nppig_LIBRARY}
    ${CUDA_nppim_LIBRARY}
    ${CUDA_nppist_LIBRARY}
    ${CUDA_nppisu_LIBRARY}
    ${CUDA_nppitc_LIBRARY}
  )
ENDIF()


# Add CMake CXX to Compilation Flags
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
include_directories( ${HEADER_PATH} )

IF (NOT NO_CUDA_BUILD)
  add_executable (CSFT csft_driver.cpp config.cpp csft_det.cpp fed.cpp nldiffusion_functions.cpp akaze_utils.cpp csft_desc.cpp ncc_gpu.cpp)
  target_link_libraries(CSFT ${SYSTEM_LIBS} ${CUDA_LIBRARIES} ${NPP_CUDA})
ELSE()
  add_executable (CSFT csft_driver.cpp config.cpp csft_det.cpp fed.cpp nldiffusion_functions.cpp akaze_utils.cpp csft_desc.cpp ncc_cpu.cpp)
  target_link_libraries(CSFT ${SYSTEM_LIBS})
ENDIF()

# Set the USE_CUDA flag for use within the actual code
IF (NOT NO_CUDA_BUILD)
  target_compile_definitions(CSFT PRIVATE USE_CUDA=1)
ELSE()
  target_compile_definitions(CSFT PRIVATE USE_CUDA=0)
ENDIF()
