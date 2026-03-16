# Running Experiments

## Environment

The repository includes a project-local virtual environment under `.venv/`.

Run scripts from the repository root:

```powershell
.\.venv\Scripts\python.exe .\experiments\6d\six_dof.py
```

## Recommended Entry Points

### Baseline 6D run

```powershell
.\.venv\Scripts\python.exe .\experiments\6d\six_dof.py
```

### 6D excited-state run

```powershell
.\.venv\Scripts\python.exe .\experiments\6d\six_dof_excited.py
```

### Representative 5D run

```powershell
.\.venv\Scripts\python.exe .\experiments\5d\5dofv4.py
```

## Practical Advice

- Start by shrinking walker counts and step counts when validating refactors.
- Keep long-running exploratory experiments in their dimensionality folder instead of copying them back to the repo root.
- Save large generated figures and temporary experiment outputs outside version control, or add them to `.gitignore` before committing.

## Smoke-Test Pattern

For code validation, temporarily reduce values such as:

- `N_walkers`
- `N_burn_in`
- `N_steps`

Then run a small experiment and confirm:

- the script starts,
- tensors allocate on the expected device,
- MH sampling runs,
- the loss backpropagates,
- logs print without crashing.

## Notes On Historical Scripts

Some files in `archive/legacy/` may still run, but they are retained for reference rather than daily use. Prefer the scripts in `experiments/` for new work.
