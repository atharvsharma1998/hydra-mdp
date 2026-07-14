# JMLR MLOSS — what to submit and where

Rules: https://www.jmlr.org/mloss/mloss-info.html

## Portal upload (three pieces + link)

| Item | File / link |
|------|-------------|
| Cover letter (must say **MLOSS track**) | `jmlr/cover_letter.pdf` |
| Paper (≤4 pages + refs) | `jmlr/bevfusion_planner.pdf` |
| Public repository | `https://github.com/atharvsharma1998/hydra-mdp` @ tag `v0.1.0-mloss` |
| Source archive of the reviewed version | `jmlr/submission/v0.1.0-mloss.tar.gz` (rebuild with `bash jmlr/pack_submission.sh`) |

After acceptance they also host a code archive on jmlr.org (name like `sharma26a-code.tar.gz`).

## What goes in the **code tarball** (source)

Include:

- Python training / eval / export code
- C++ `deploy/` sources and build scripts
- Docs: `README.md`, `INSTALL.md`, `docs/*`, `NOTICE`, `LICENSE`
- Demo GIF under `assets/demo/`
- Paper TeX under `jmlr/` (optional but useful)

**Do not** put in the tarball:

- Multi-GB NAVSIM sensor data
- PyTorch `.pth` (~488 MB) — put on **GitHub Release**
- ONNX graphs (~175 MB) — put on **GitHub Release**
- TensorRT `.plan` (GPU-specific; users rebuild)
- `.git/`, `__pycache__/`, credentials, parity binary dumps if huge

Release assets are documented in [`docs/MODELS.md`](../docs/MODELS.md).

## What reviewers expect to find on GitHub

1. Clear README (SOPHI, not an unrelated fork landing page)
2. `INSTALL.md` + deploy / train docs with copy-paste commands
3. Tagged version matching the cover letter
4. License (Apache-2.0) + `NOTICE` for CUDA/TensorRT / data boundaries
5. Issues enabled
6. (Recommended) Release with checkpoint + ONNX zip

## Before submit

- [ ] Repo public; About text describes SOPHI
- [ ] Tag `v0.1.0-mloss` pushed
- [ ] Release assets: `.pth` + `sophi_onnx_navtrain_v1.tar.gz`; update URLs in `docs/MODELS.md`
- [ ] Cover letter: real reviewer names; OSI license; URL; version; community note
- [ ] Rebuild tarball: `bash jmlr/pack_submission.sh`
- [ ] Spot-check docs for leftover draft wording

## Suggested upload order

1. Push docs + tag + Release weights  
2. Fill cover-letter reviewers  
3. Rebuild PDFs if needed  
4. Submit via https://jmlr.csail.mit.edu/manudb/
