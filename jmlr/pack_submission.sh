#!/usr/bin/env bash
# Build a JMLR MLOSS source archive (no .git, no checkpoints, no multi-GB data).
# Includes deploy/example-data/ (one frame) for clone-and-run C++ inference.
# Archive name follows JMLR convention: <author><yy><letter>-code.tar.gz
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Support both layouts:
#   hydramdp/jmlr  + hydramdp/navsim
#   navsim/jmlr    (paper packaged inside the git repo)
if [ -d "$ROOT/navsim/navsim" ] || [ -f "$ROOT/navsim/setup.py" ]; then
  NAVSIM="$ROOT/navsim"
  PAPER="$ROOT/jmlr"
elif [ -f "$ROOT/setup.py" ] && [ -d "$ROOT/navsim" ]; then
  NAVSIM="$ROOT"
  PAPER="$ROOT/jmlr"
else
  echo "ERROR: cannot locate navsim repo root from $ROOT"; exit 1
fi
OUT_DIR="$PAPER/submission"
STAGE_NAME="sharma26a-code"
STAGE="$OUT_DIR/$STAGE_NAME"
TAG="v0.1.0-mloss"
ARCHIVE="$OUT_DIR/${STAGE_NAME}.tar.gz"

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
  --exclude 'jmlr/*.aux' \
  --exclude 'jmlr/*.log' \
  --exclude 'jmlr/*.bbl' \
  --exclude 'jmlr/*.blg' \
  --exclude 'jmlr/*.out' \
  --exclude '*.pth' \
  --exclude '*.onnx' \
  --exclude '*.plan' \
  --exclude '*.npy' \
  --exclude '.env' \
  --exclude '**/*credentials*' \
  --exclude '**/*.pt' \
  "$NAVSIM/" "$STAGE/"

# Paper sources + figures (overwrite any stale copies)
mkdir -p "$STAGE/jmlr/figures"
cp -a "$PAPER/bevfusion_planner.tex" \
      "$PAPER/bevfusion_planner.bib" \
      "$PAPER/jmlr2e.sty" \
      "$PAPER/cover_letter.tex" \
      "$PAPER/LICENSE" \
      "$STAGE/jmlr/"
cp -a "$PAPER/figures/architecture_pipeline.png" \
      "$PAPER/figures/latency_breakdown.pdf" \
      "$PAPER/figures/latency_breakdown.png" \
      "$STAGE/jmlr/figures/" 2>/dev/null || true

# Top-level license notices from the repo (not the jmlr stub)
cp -a "$NAVSIM/LICENSE" "$STAGE/LICENSE"
cp -a "$NAVSIM/NOTICE" "$STAGE/NOTICE"
cp -a "$NAVSIM/NOTICE" "$STAGE/jmlr/NOTICE"

# Version stamp
cat > "$STAGE/VERSION" <<EOF
$TAG
project=https://github.com/atharvsharma1998/hydra-mdp
branch=main
models=https://drive.google.com/drive/folders/1hcnOaJvWhL3hzBxSSsUlCLYQ8InSOYGE?usp=sharing
EOF

# Sanity checks
echo "Largest staged files:"
find "$STAGE" -type f -printf '%s %p\n' | sort -nr | head -15 || true
test -f "$STAGE/deploy/example-data/points.tensor" || { echo "ERROR: missing example-data"; exit 1; }
test -f "$STAGE/QUICKSTART.md" || { echo "ERROR: missing QUICKSTART.md"; exit 1; }
test -f "$STAGE/docs/MODELS.md" || { echo "ERROR: missing docs/MODELS.md"; exit 1; }
grep -q 'drive.google.com' "$STAGE/docs/MODELS.md" || { echo "ERROR: Drive link missing in MODELS.md"; exit 1; }
if grep -qi 'sophi' "$STAGE/README.md"; then
  echo "ERROR: SOPHI still in README"; exit 1
fi

tar -C "$OUT_DIR" -czf "$ARCHIVE" "$STAGE_NAME"
echo
echo "Wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
echo "Contents root: $STAGE_NAME/"
echo
echo "Upload trio for JMLR portal:"
echo "  1) $PAPER/cover_letter.pdf"
echo "  2) $PAPER/bevfusion_planner.pdf"
echo "  3) $ARCHIVE"
echo "  + public repo: https://github.com/atharvsharma1998/hydra-mdp  (branch main)"
