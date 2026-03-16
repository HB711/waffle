# Project Structure

## Why This Layout Exists

The original project grew as a flat directory of experiment snapshots. That is normal for early research work, but it makes it hard to tell:

- which files are active,
- which files are analysis-only,
- which files are historical copies,
- which scripts share logic.

The current layout keeps the code history intact while adding a clearer operational structure.

## Top-Level Directories

### `experiments/`

Active or still-useful research scripts, grouped by dimensionality.

- `experiments/4d/`: early 4D VMC experiments.
- `experiments/5d/`: 5D experiments, including local sampling, explicit-gradient variants, and sweep scripts.
- `experiments/6d/`: 6D experiments and the extracted 6D shared core.

### `analysis/`

Scripts and outputs used to inspect the physical landscape rather than train wavefunctions directly.

- `analysis/potentials/`: potential scans, minima, and saddle analysis.
- `analysis/reports/`: exported tables and TeX summaries.

### `archive/legacy/`

Historical copies that should not be mistaken for the current mainline.

This includes:

- `_copy` variants,
- GPT-generated drafts,
- ad hoc historical snapshots with ambiguous names.

### `docs/`

Human-facing documentation for navigation, running experiments, and maintenance.

## Active 6D Path

The clearest maintained 6D path is:

1. `experiments/6d/six_dof_core.py`
2. `experiments/6d/six_dof.py`
3. `experiments/6d/six_dof_excited.py`

`six_dof_core.py` now contains the shared 6D physical model, neural ansatz, MH sampler, local-energy computation, and stable overlap utility. This removes the most obvious duplication from the 6D family.

## Active 5D Path

The 5D folder is still more exploratory than the 6D folder. Good entry points are:

- `experiments/5d/5dofv4.py`
- `experiments/5d/314.py`
- `experiments/5d/sequence.py`
- `experiments/5d/local.py`
- `experiments/5d/GS.py`

These represent different strategies rather than a single polished framework, so the folder is intentionally grouped but not heavily normalized yet.

## What Is Still Messy

- Many experiment scripts still duplicate physics setup and local-energy code.
- Output paths are not yet centralized across all scripts.
- File names reflect experiment history rather than a strict naming convention.
- The 5D family still contains several near-neighbor variants that could be merged later.

## Recommended Next Refactors

- Extract a shared 5D core similar to `experiments/6d/six_dof_core.py`.
- Standardize output directories for plots and exported tables.
- Rename selected legacy scripts to clearer descriptive names before adding more experiments.
- Add smoke-test entry points for short verification runs.
