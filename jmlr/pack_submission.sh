#!/usr/bin/env bash
# Build a JMLR MLOSS source archive (no .git, no checkpoints, no multi-GB data).
# Run from anywhere; writes jmlr/submission/sophi_v0.1.0-mloss.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Support both layouts:
#   hydramdp/jmlr  + hydramdp/navsim
#   navsim/jmlr    (paper packaged inside the git repo)
if [ -d "$ROOT/navsim/navsim" ] || [ -f "$ROOT/navsim/setup.py" ]; then
  NAVSIM="$ROOT/navsim"
elif [ -f "$ROOT/setup.py" ] && [ -d "$ROOT/navsim" ]; then
  NAVSIM="$ROOT"
else
  echo "ERROR: cannot locate navsim repo root from $ROOT"; exit 1
fi
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submission"
STAGE="$OUT_DIR/sophi_v0.1.0-mloss"
TAG="v0.1.0-mloss"
ARCHIVE="$OUT_DIR/${TAG}.tar.gz"

rm -rf "$STAGE"
mkdir -p "$STAGE"

echo "Staging from $NAVSIM -> $STAGE"

# Core source (exclude heavy / generated / private paths)
rsync -a \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude 'navsim.egg-info/' \
  --exclude 'navsim_workspace/' \
  --exclude '**/metric_cache/' \
  --exclude '**/teacher_scores_cache*/' \
  --exclude '**/tb/' \
  --exclude 'assets/navsim_*.png' \
  --exclude 'assets/navsim_*.gif' \
  --exclude 'assets/navsimv2_*' \
  --exclude 'tutorial/' \
  --exclude 'traj_final/' \
  --exclude 'deploy/build/' \
  --exclude 'deploy/tool/simhei.ttf' \
  --exclude 'deploy/model/**/build/*.plan' \
  --exclude 'deploy/model/**/build/*.log' \
  --exclude 'deploy/model/**/build/*.json' \
  --exclude 'deploy/parity-data/cpp/' \
  --exclude 'deploy/parity-data/ref/' \
  --exclude 'deploy/parity-data/*.tensor' \
  --exclude 'jmlr/submission/' \
  --exclude '**/submission/' \
  --exclude '*.pth' \
  --exclude '*.onnx' \
  --exclude '*.plan' \
  --exclude '*.npy' \
  --exclude '.env' \
  --exclude '**/*credentials*' \
  --exclude '**/*.pt' \
  "$NAVSIM/" "$STAGE/"

# Paper + cookbook + license notices
mkdir -p "$STAGE/jmlr"
cp -a "$ROOT/jmlr/bevfusion_planner.tex" \
      "$ROOT/jmlr/bevfusion_planner.bib" \
      "$ROOT/jmlr/jmlr2e.sty" \
      "$ROOT/jmlr/cover_letter.tex" \
      "$ROOT/jmlr/REPRODUCIBILITY.md" \
      "$ROOT/jmlr/README.md" \
      "$ROOT/jmlr/LICENSE" \
      "$ROOT/jmlr/NOTICE" \
      "$STAGE/jmlr/"
mkdir -p "$STAGE/jmlr/figures"
cp -a "$ROOT/jmlr/figures/architecture_pipeline.png" \
      "$ROOT/jmlr/figures/latency_breakdown.pdf" \
      "$ROOT/jmlr/figures/latency_breakdown.png" \
      "$STAGE/jmlr/figures/"

# Ensure top-level NOTICE/LICENSE present
cp -a "$ROOT/jmlr/LICENSE" "$STAGE/LICENSE"
cp -a "$ROOT/jmlr/NOTICE" "$STAGE/NOTICE"

# Version stamp
cat > "$STAGE/VERSION" <<EOF
$TAG
project=https://github.com/atharvsharma1998/hydra-mdp
EOF

# Sanity: no huge files
echo "Largest staged files:"
find "$STAGE" -type f -printf '%s %p\n' | sort -nr | head -15 || true

tar -C "$OUT_DIR" -czf "$ARCHIVE" "$(basename "$STAGE")"
echo "Wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
echo
echo "Next: git tag and push (from $NAVSIM):"
echo "  git tag -a $TAG -m 'JMLR MLOSS release $TAG'"
echo "  git push origin $TAG"
echo "  git push origin HEAD   # if paper/docs commits are not yet on remote"
