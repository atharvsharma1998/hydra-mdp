# GTRS-BEVFusion C++ deploy pipeline (published with NAVSIM).
#
# Authored here: src/gtrs/* (camera/lidar front-ends, head decoders, orchestrator,
# entry point). The heavy BEVFusion *core* (camera/lidar/fuser CUDA modules,
# custom spconv ONNX parser, TensorRT wrapper, nv::Tensor, cuOSD, roiconvert)
# is reused from the CUDA-BEVFusion fork via BEVFUSION_ROOT -- we link its
# prebuilt libbevfusion_core.so rather than vendoring it.
#
# Build (after building libbevfusion_core.so once in the fork):
#   mkdir build && cd build
#   cmake .. -DBEVFUSION_ROOT=/home/atharv/Lidar_AI_Solution/CUDA-BEVFusion \
#            -DTensorRT_Root=/usr/local/TensorRT-8.6.1.6 -DCUDASM=86
#   make -j
cmake_minimum_required(VERSION 2.8.12)
project(gtrs_bevfusion_deploy)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")

# ---- locate the BEVFusion core + sibling libraries/dependencies ----
if(NOT DEFINED BEVFUSION_ROOT)
  if(DEFINED ENV{BEVFUSION_ROOT})
    set(BEVFUSION_ROOT $ENV{BEVFUSION_ROOT})
  else()
    set(BEVFUSION_ROOT "/home/atharv/Lidar_AI_Solution/CUDA-BEVFusion" CACHE PATH "CUDA-BEVFusion fork root")
  endif()
endif()
get_filename_component(AI_SOLUTION_ROOT "${BEVFUSION_ROOT}/.." ABSOLUTE)
set(LIB_ROOT "${AI_SOLUTION_ROOT}/libraries")
set(DEPS_ROOT "${AI_SOLUTION_ROOT}/dependencies")
set(BEVFUSION_BUILD "${BEVFUSION_ROOT}/build" CACHE PATH "dir containing libbevfusion_core.so")

# ---- CUDA / TensorRT / spconv ----
if(NOT DEFINED CUDA_HOME)
  set(CUDA_HOME "$ENV{CUDA_HOME}")
  if("${CUDA_HOME}" STREQUAL "")
    set(CUDA_HOME "/usr/local/cuda")
  endif()
endif()
set(CUDA_Lib "${CUDA_HOME}/lib64")
set(CUDA_Inc "${CUDA_HOME}/include")

if(NOT DEFINED TensorRT_Root)
  set(TensorRT_Root "$ENV{TensorRT_Root}")
  if("${TensorRT_Root}" STREQUAL "")
    set(TensorRT_Root "/usr/local/TensorRT-8.6.1.6")
  endif()
endif()
set(TensorRT_Inc "${TensorRT_Root}/include")
set(TensorRT_Lib "${TensorRT_Root}/lib")

if(NOT DEFINED CUDASM)
  set(CUDASM "$ENV{CUDASM}")
  if("${CUDASM}" STREQUAL "")
    set(CUDASM "86")
  endif()
endif()

if(DEFINED ENV{SPCONV_ROOT} AND NOT "$ENV{SPCONV_ROOT}" STREQUAL "")
  set(SPCONV_BASE "$ENV{SPCONV_ROOT}")
else()
  set(SPCONV_BASE "/home/atharv/spconv")
endif()
set(spconv_include ${SPCONV_BASE}/example/libspconv/spconv/include)
set(spconv_lib ${SPCONV_BASE}/example/libspconv/build/spconv/src)
set(cumm_include ${SPCONV_BASE}/example/libspconv/cumm/include)

# Pin to CUDA 11.4 (matches libbevfusion_core.so + libspconv ABI). Without this,
# FindCUDA auto-detects a stray system CUDA (e.g. 11.5 under /usr).
set(CUDA_TOOLKIT_ROOT_DIR "${CUDA_HOME}" CACHE PATH "" FORCE)
set(CUDA_BIN_PATH "${CUDA_HOME}/bin")
# nvcc 11.4 supports host gcc <= 10; the system default is gcc 11. Use g++-10 for
# the CUDA host side (override with -DCUDA_HOST_COMPILER=...).
if(NOT DEFINED CUDA_HOST_COMPILER)
  if(EXISTS "/usr/bin/g++-10")
    set(CUDA_HOST_COMPILER "/usr/bin/g++-10")
  endif()
endif()
find_package(CUDA REQUIRED)
if(NOT CUDA_VERSION VERSION_EQUAL "11.4")
  message(WARNING "Expected CUDA 11.4 (got ${CUDA_VERSION}); ensure CUDA_HOME points at /usr/local/cuda-11.4")
endif()

# protobuf is only used by the prebuilt core (custom pb 3.12.4, resolved via the
# core .so's NEEDED/rpath); our sources don't need it. Link only if explicitly found.
find_package(Protobuf QUIET)

# Match the core's ABI (CUDA 11.x -> c++14 + tensorview hw-acc defines).
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++14 -O3 -Wno-deprecated-declarations -DENABLE_TEXT_BACKEND_STB -DTV_ENABLE_HARDWARE_ACC -DTV_HARDWARE_ACC_CUDA")
# Apply build-type-independently too: if CMAKE_BUILD_TYPE is unset, *_RELEASE flags
# are skipped and cuOSD's StbTrueType text backend (.cpp) would compile out, losing
# all on-image labels. add_definitions guarantees the host .cpp side gets it.
add_definitions(-DENABLE_TEXT_BACKEND_STB)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_${CUDASM},code=sm_${CUDASM}
    -DENABLE_TEXT_BACKEND_STB -DTV_ENABLE_HARDWARE_ACC -DTV_HARDWARE_ACC_CUDA)
if(CUDA_HOST_COMPILER)
  list(APPEND CUDA_NVCC_FLAGS -ccbin ${CUDA_HOST_COMPILER})
endif()

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/src        # our headers ("gtrs/...")
  ${BEVFUSION_ROOT}/src                  # "bevfusion/...", "common/..."
  ${BEVFUSION_ROOT}/src/common           # visualize.cu -> "dtype.hpp"
  ${LIB_ROOT}/cuOSD/src
  ${LIB_ROOT}/roiconvert
  ${DEPS_ROOT}/stb
  ${spconv_include}
  ${cumm_include}
  ${CUDA_Inc}
  ${TensorRT_Inc}
)

link_directories(
  ${CUDA_Lib}
  ${TensorRT_Lib}
  ${spconv_lib}
  ${BEVFUSION_BUILD}
)

file(GLOB GTRS_FILES src/gtrs/*.cu src/gtrs/*.cpp)
file(GLOB_RECURSE CUOSD_FILES ${LIB_ROOT}/cuOSD/src/*.cpp ${LIB_ROOT}/cuOSD/src/*.cu)

cuda_add_executable(gtrs_bevfusion
  ${GTRS_FILES}
  ${LIB_ROOT}/roiconvert/roi_conversion/roi_conversion.cu
  ${BEVFUSION_ROOT}/src/common/visualize.cu
  ${CUOSD_FILES}
)

target_link_libraries(gtrs_bevfusion
  ${BEVFUSION_BUILD}/libbevfusion_core.so
  ${spconv_lib}/libspconv.so
  libnvinfer.so
  libnvinfer_plugin.so
  libcudart.so
  libcublasLt.so
)
