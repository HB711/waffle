# Waffle Excited States

Research code for variational Monte Carlo experiments on waffle-style superconducting phase models.

This repository has been reorganized to separate active experiments, analysis scripts, and archived variants. The goal is to make the codebase easier to navigate without rewriting every historical experiment into a single framework.

## Repository Layout

```text
.
|-- experiments/
|   |-- 4d/
|   |-- 5d/
|   `-- 6d/
|-- analysis/
|   |-- potentials/
|   `-- reports/
|-- archive/
|   `-- legacy/
|-- docs/
|-- .venv/
`-- README.md
```

## Where To Start

- `experiments/6d/six_dof.py`: current 6D baseline VMC run.
- `experiments/6d/six_dof_excited.py`: current 6D multi-state / excited-state workflow.
- `experiments/6d/six_dof_core.py`: shared 6D core logic extracted from the earlier flat scripts.
- `experiments/5d/5dofv4.py`: one of the more mature 5D tempered multi-state experiments.
- `experiments/5d/sequence.py`: staged 5D training flow for ground and excited states.
- `analysis/potentials/`: potential-surface and saddle-point analysis utilities.

## Quick Start

Run experiments from the repository root so outputs land in a predictable working directory.

```powershell
.\.venv\Scripts\python.exe .\experiments\6d\six_dof.py
.\.venv\Scripts\python.exe .\experiments\6d\six_dof_excited.py
.\.venv\Scripts\python.exe .\experiments\5d\5dofv4.py
```

## Engineering Conventions

- New experimental scripts should go under `experiments/<dimension>/`.
- Shared helpers should be factored next to the experiments that use them, or promoted into a dedicated shared module when they are stable.
- One-off analysis scripts belong in `analysis/`.
- Historical copies, GPT-generated variants, and superseded snapshots belong in `archive/legacy/`.
- Avoid adding new top-level `.py` files unless they are true repository entry points.

## Documentation

- `docs/PROJECT_STRUCTURE.md`: detailed project map and rationale for the current layout.
- `docs/RUNNING_EXPERIMENTS.md`: how to run experiments and how to keep outputs manageable.
- `experiments/5d/README.md`: notes for the 5D experiment family.
- `experiments/6d/README.md`: notes for the 6D experiment family.

## Current Status

This repository is still a research codebase rather than a polished library. The reorganization focuses on:

- reducing root-directory clutter,
- separating active code from legacy snapshots,
- documenting the main paths through the project,
- preserving historical scripts without pretending they are all production-ready.
