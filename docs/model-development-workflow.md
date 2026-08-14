# Model Development Workflow

The shared repository is the source of truth. Model work starts only after the
latest remote `main` has been incorporated.

## Before coding

```text
git status --short --branch
git fetch origin
git switch main
git pull --ff-only origin main
git switch <feature-branch>
git merge main
```

Do not begin with unresolved conflicts or an uncommitted shared worktree. New
branches should be created from the freshly pulled `main`.

## Project boundary

- Production, UI-independent functionality belongs in `src/core/`.
- Controllers import the stable surface with `from core import ...`.
- Feature implementations use `<feature>_model.py`, for example
  `visualization_model.py`, `calibration_model.py`, and
  `super_resolution_model.py`.
- Shared infrastructure keeps domain names such as `hsi_data.py`,
  `hsi_reader.py`, and `errors.py`.
- Model code must not import PyQt6, PySide6, or application View classes.

Local test code, sample captures, and the temporary Model test UI are ignored
by Git. They are development aids and are not part of the production upload.

## Verification before pushing

Run tests locally in `comp_6730`, confirm `git status` contains only intended
production files, and check ignored assets with:

```text
git status --short --ignored
```
