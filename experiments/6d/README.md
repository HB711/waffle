# 6D Experiments

This folder contains the cleaner 6D branch of the repository.

## Main Files

- `six_dof_core.py`: shared 6D model, sampler, local-energy, and overlap utilities.
- `six_dof.py`: baseline 6D VMC training script.
- `six_dof_excited.py`: multi-state 6D training script.

## Additional Variants

- `six_dof_ex.py`
- `six_dof_local.py`
- `six_dof_stable.py`
- `6_dof_v2.py`

These are retained as alternative experiments or historical stepping stones, but the preferred starting point is the `six_dof.py` plus `six_dof_core.py` path.
