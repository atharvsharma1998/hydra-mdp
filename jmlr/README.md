# JMLR MLOSS package — SOPHI

LaTeX + reproducibility materials for the JMLR **Machine Learning Open Source
Software (MLOSS)** track submission.

Manuscript: *SOPHI: An Open End-to-End Camera--LiDAR Planning Stack with
Sparse-Convolution Inference for NAVSIM*

## Track

This package targets **[JMLR MLOSS](https://www.jmlr.org/mloss/mloss-info.html)**
(≤4 pages + references), not the main research track. A longer research draft is
kept as `bevfusion_planner_research_draft.tex` for reference only.

## Build PDF

Minimal Debian/Ubuntu TeX installs may miss `pdftexcmds.sty`. Either install:

```bash
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended
```

Or keep `nohyperref` in `bevfusion_planner.tex` (already set).

```bash
cd /home/atharv/Downloads/hydramdp/jmlr
pdflatex bevfusion_planner.tex
bibtex bevfusion_planner
pdflatex bevfusion_planner.tex
pdflatex bevfusion_planner.tex
pdflatex cover_letter.tex
```

Requires `jmlr2e.sty` (included) and `natbib` (from `texlive-latex-recommended`).

**Page budget:** body ≤ 4 pages; references may continue. Portal PDF should stay under 5 MB.

## Files

| File | Purpose |
|------|---------|
| `bevfusion_planner.tex` | MLOSS paper (JMLR style) |
| `bevfusion_planner_research_draft.tex` | Previous long research draft (not for MLOSS submit) |
| `bevfusion_planner.bib` | Bibliography |
| `jmlr2e.sty` | Official style file |
| `cover_letter.tex` | MLOSS cover letter |
| `REPRODUCIBILITY.md` | Train / eval / export / TRT / C++ cookbook |
| `NOTICE` | Third-party and proprietary-dependency notice |
| `LICENSE` | Apache-2.0 (mirror of repo license) |
| `figures/` | Architecture + latency figures |
| `pack_submission.sh` | Build code archive for the portal |

## Before submission

1. Confirm public GitHub tag `v0.1.0-mloss` on https://github.com/atharvsharma1998/hydra-mdp
2. Fill concrete AE / reviewer names in the cover letter portal fields
3. Run `./pack_submission.sh` and verify the archive builds without secrets/checkpoints
4. Compile PDFs; check page count ≤ 4 for the paper body
5. Use `[preprint]` until acceptance; switch camera-ready `\jmlrheading{...}` per the [author guide](https://www.jmlr.org/format/authors-guide.html)

## Source code policy

MLOSS **requires** open-source code under a recognized OSI license (Apache-2.0 here),
a public repository URL, a version tag, and a source archive. See `REPRODUCIBILITY.md`
and `NOTICE` for CUDA/TensorRT and NAVSIM data boundaries.
