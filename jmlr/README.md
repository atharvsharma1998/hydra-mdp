# MLOSS software description

Short JMLR MLOSS write-up for the open-source release:
**Sparse-conv Offline Perception with Hydra-MDP Inference**.

The project itself lives on `main` at
https://github.com/atharvsharma1998/hydra-mdp — this folder is only the
four-page software description + cover letter.

## Build PDF

```bash
cd jmlr
pdflatex bevfusion_planner.tex
bibtex bevfusion_planner
pdflatex bevfusion_planner.tex
pdflatex bevfusion_planner.tex
pdflatex cover_letter.tex
```

## Pack source archive

```bash
bash pack_submission.sh
# → submission/v0.1.0-mloss.tar.gz  (includes deploy/example-data/)
```

User-facing deploy docs (not MLOSS-specific): repo root `QUICKSTART.md`.
