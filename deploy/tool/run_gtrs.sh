#!/bin/bash
# Build + run the GTRS-BEVFusion C++ deploy pipeline (lives in the NAVSIM repo).
#
#   BEVFUSION_ROOT=/home/atharv/Lidar_AI_Solution/CUDA-BEVFusion \
#   GTRS_ONNX_DIR=/home/atharv/Downloads/hydramdp/navsim_workspace/onnx \
#   GTRS_DATA=/path/to/example-data \
#   bash deploy/tool/run_gtrs.sh [fp16|int8]
set -e

PRECISION=${1:-fp16}
BEVFUSION_ROOT=${BEVFUSION_ROOT:-/home/atharv/Lidar_AI_Solution/CUDA-BEVFusion}
TENSORRT_ROOT=${TensorRT_Root:-/usr/local/TensorRT-8.6.1.6}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
SPCONV_ROOT=${SPCONV_ROOT:-/home/atharv/spconv}
CUDASM=${CUDASM:-86}
MODEL=${DEBUG_MODEL:-gtrs_bevfusion}

DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA=${GTRS_DATA:-$DEPLOY_DIR/example-data}

# 0) the core must be built once in the fork
CORE="$BEVFUSION_ROOT/build/libbevfusion_core.so"
if [ ! -f "$CORE" ]; then
  echo "ERROR: $CORE not found."
  echo "Build it once in the fork:  (cd $BEVFUSION_ROOT && bash tool/run.sh)  or its cmake/make."
  exit 1
fi

# 1) TensorRT engines
DEBUG_MODEL=$MODEL bash "$DEPLOY_DIR/tool/build_gtrs_engines.sh" "$PRECISION"

# 2) build the C++ target
mkdir -p "$DEPLOY_DIR/build"
cd "$DEPLOY_DIR/build"
cmake .. -DBEVFUSION_ROOT="$BEVFUSION_ROOT" -DTensorRT_Root="$TENSORRT_ROOT" \
         -DCUDA_HOME="$CUDA_HOME" -DCUDASM="$CUDASM"
make gtrs_bevfusion -j

# 3) run (from deploy root so model/<tag> + data resolve)
cd "$DEPLOY_DIR"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$TENSORRT_ROOT/lib:$SPCONV_ROOT/example/libspconv/build/spconv/src:$BEVFUSION_ROOT/build:$LD_LIBRARY_PATH"
./build/gtrs_bevfusion "$DATA" "$MODEL" "$PRECISION"
