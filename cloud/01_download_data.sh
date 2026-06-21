#!/bin/bash
# Download maps + trainval logs + full navtrain sensor blobs (32 current + 32
# history shards, ~425 GB extracted) into $NAVSIM_WS on the network volume.
# Resumable: completed shards are marked and skipped; wget -c resumes partials.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

HF=https://huggingface.co/datasets/OpenDriveLab/OpenScene/resolve/main
mkdir -p "$NAVSIM_WS" && cd "$NAVSIM_WS"
mkdir -p .dl_marks trainval_sensor_blobs/trainval

echo "=== disk free on volume ==="; df -h "$NAVSIM_WS" | tail -1

# --- 1. maps ----------------------------------------------------------------
if [ ! -d "$NAVSIM_WS/maps" ]; then
  echo "=== maps ==="
  wget -c -O nuplan-maps-v1.1.zip https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-maps-v1.1.zip
  unzip -q nuplan-maps-v1.1.zip && rm nuplan-maps-v1.1.zip && mv nuplan-maps-v1.0 maps
else echo "maps: present, skip"; fi

# --- 2. trainval logs -------------------------------------------------------
if [ ! -d "$NAVSIM_WS/trainval_navsim_logs" ]; then
  echo "=== trainval logs ==="
  wget -c -O meta.tgz "$HF/openscene-v1.1/openscene_metadata_trainval.tgz"
  tar -xzf meta.tgz && rm meta.tgz
  mv openscene-v1.1/meta_datas trainval_navsim_logs && rm -rf openscene-v1.1
else echo "trainval logs: present, skip"; fi

# --- 3. navtrain sensor shards ---------------------------------------------
fetch_shard() {  # $1=current|history  $2=split index
  local kind=$1 split=$2
  local mark=".dl_marks/${kind}_${split}.done"
  local tgz="navtrain_${kind}_${split}.tgz"
  local dir="navtrain_${kind}_${split}"   # HF tarballs extract to this dir name
  if [ -f "$mark" ]; then echo "[$kind $split] done, skip"; return; fi
  echo "[$kind $split] downloading"
  wget -c -O "$tgz" "$HF/navsim/navtrain_${kind}_${split}.tgz" || { echo "[$kind $split] download FAILED"; return 1; }
  echo "[$kind $split] extracting"
  tar -xzf "$tgz" && rm "$tgz"
  rsync -a "$dir"/* trainval_sensor_blobs/trainval/ && rm -rf "$dir"
  touch "$mark"
  echo "[$kind $split] OK"
}

for split in $(seq 1 32); do fetch_shard current "$split"; done
for split in $(seq 1 32); do fetch_shard history "$split"; done

echo "=== done. sensor log dirs: $(ls trainval_sensor_blobs/trainval | wc -l) ==="
df -h "$NAVSIM_WS" | tail -1
