# JMLR MLOSS — what to submit

Rules: https://www.jmlr.org/mloss/mloss-info.html

This is an **open-source software** submission (MLOSS track), not a long research paper.

## Portal upload

| Item | File / link |
|------|-------------|
| Cover letter (must say **MLOSS track**) | `jmlr/cover_letter.pdf` |
| ≤4-page description | `jmlr/bevfusion_planner.pdf` |
| Public repository | `https://github.com/atharvsharma1998/hydra-mdp` (`main`) |
| Source archive | `bash jmlr/pack_submission.sh` → `submission/v0.1.0-mloss.tar.gz` |

## In the code tarball

- Source + `QUICKSTART.md` + docs
- `deploy/example-data/` (one frame for C++ smoke test)
- Demo GIF under `assets/demo/`
- This `jmlr/` folder

**Not** in the tarball: multi-GB NAVSIM data, `.pth`, ONNX zip, TensorRT `.plan`.
Those go on Google Drive ([`docs/MODELS.md`](../docs/MODELS.md)).

## Before submit

- [ ] Paste real Google Drive links into `docs/MODELS.md`
- [ ] Repo public; README describes the project (no stale upstream NAVSIM landing page)
- [ ] Tag release if desired; otherwise cover letter says `main`
- [ ] Cover letter: real reviewer names
- [ ] Rebuild PDFs + tarball
- [ ] Spot-check: no leftover acronym branding you do not want
