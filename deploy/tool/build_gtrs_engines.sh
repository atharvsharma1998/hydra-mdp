#!/bin/bash
# Build TensorRT engines for GTRS-BEVFusion from the exported ONNX graphs.
# Self-contained (no dependency on the CUDA-BEVFusion fork's environment.sh).
#
#   GTRS_ONNX_DIR=/home/atharv/Downloads/hydramdp/navsim_workspace/onnx \
#   bash deploy/tool/build_gtrs_engines.sh [fp16|int8]
#
# Stages ONNX into deploy/model/gtrs_bevfusion/ and builds the 6 TRT engines.
# The LiDAR backbone stays raw ONNX (run by the custom spconv parser at runtime).
set -e

PRECISION=${1:-${DEBUG_PRECISION:-fp16}}
TENSORRT_ROOT=${TensorRT_Root:-/usr/local/TensorRT-8.6.1.6}
TRTEXEC=${TRTEXEC:-$TENSORRT_ROOT/bin/trtexec}
GTRS_ONNX_DIR=${GTRS_ONNX_DIR:-/home/atharv/Downloads/hydramdp/navsim_workspace/onnx}

# resolve deploy root (this script lives in deploy/tool/)
DEPLOY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL=${DEBUG_MODEL:-gtrs_bevfusion}
base="$DEPLOY_DIR/model/$MODEL"
build="$base/build"
mkdir -p "$build"

[ -x "$TRTEXEC" ] || { echo "trtexec not found at $TRTEXEC (set TensorRT_Root or TRTEXEC)"; exit 1; }
export LD_LIBRARY_PATH="$TENSORRT_ROOT/lib:${LD_LIBRARY_PATH}"

echo "Staging ONNX from $GTRS_ONNX_DIR -> $base"
for f in camera.backbone camera.vtransform fuser planning_head det_head seg_head; do
  src="$GTRS_ONNX_DIR/$f.onnx"
  [ -f "$src" ] || { echo "MISSING $src"; exit 1; }
  cp -u "$src" "$base/$f.onnx"
done
if [ -f "$GTRS_ONNX_DIR/lidar.backbone.onnx" ]; then
  cp -u "$GTRS_ONNX_DIR/lidar.backbone.onnx" "$base/lidar.backbone.xyz.onnx"
elif [ -f "$GTRS_ONNX_DIR/lidar.backbone.xyz.onnx" ]; then
  cp -u "$GTRS_ONNX_DIR/lidar.backbone.xyz.onnx" "$base/lidar.backbone.xyz.onnx"
else
  echo "WARN: no lidar.backbone(.xyz).onnx in $GTRS_ONNX_DIR (SCN will fail at runtime)"
fi

fp16_flags="--fp16"
dyn_flags="--fp16"
[ "$PRECISION" == "int8" ] && dyn_flags="--fp16 --int8"

# $1 name  $2 precision_flags  $3 n_inputs  $4 n_outputs
compile() {
  local name=$1 pflags=$2 nin=$3 nout=$4
  local onnx="$base/$name.onnx" plan="$build/$name.plan"
  [ -f "$onnx" ] || { echo "skip $name (no onnx)"; return; }
  if [ -f "$plan" ]; then echo "$plan already built"; return; fi
  local inf="--inputIOFormats=" outf="--outputIOFormats="
  for i in $(seq 1 "$nin");  do inf+=fp16:chw,;  done
  for i in $(seq 1 "$nout"); do outf+=fp16:chw,; done
  inf=${inf%?}; outf=${outf%?}
  echo "Building $plan ..."
  "$TRTEXEC" --onnx="$onnx" $pflags "$inf" "$outf" \
    --saveEngine="$plan" --memPoolSize=workspace:2048 \
    --profilingVerbosity=detailed --exportLayerInfo="$build/$name.json" \
    > "$build/$name.log" 2>&1 || { echo "FAILED $plan (see $build/$name.log)"; exit 1; }
  echo "  OK $plan"
}

# camera.backbone (ResNet50) overflows in pure fp16 -> all-NaN features. Build it
# with fp32 COMPUTE but keep fp16 IO (the fork's BEVPool/Backbone wrappers require
# fp16 feature/depth buffers). Empty precision flags => fp32 math, fp16 chw IO.
compile camera.backbone   ""           2 2
compile camera.vtransform "$fp16_flags" 1 1
compile fuser             "$dyn_flags"  2 1
compile planning_head     "$fp16_flags" 2 2
# CenterPoint det head: 1 input (fenv) -> 4 dense maps (heatmap, offset, size, heading)
compile det_head          "$fp16_flags" 1 4
compile seg_head          "$fp16_flags" 1 1

echo "Engines built under $build ; SCN runs from $base/lidar.backbone.xyz.onnx"
