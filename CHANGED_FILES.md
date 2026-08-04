# Files to update in your GitHub project

Two update rounds since the version you last pushed. If you are unsure how
far behind you are, use the checksum manifest instead of the lists:

```bash
md5sum -c MANIFEST.md5      # from the repo root; any FAILED line = replace
```

## Round 2 (most recent — the c_bound / axis-crossing fix)

Replace these 7 files:

| file | what changed |
|---|---|
| `nssr/networks.py` | `ParamNet` default `c_bound` 2.0 -> **1.0** |
| `nssr/train.py` | `train(..., c_bound=1.0)` passed to `ParamNet` |
| `nssr/metrics.py` | **new** `axis_clearance()` diagnostic |
| `scripts/train_model.py` | **new** `--c_bound` flag |
| `scripts/evaluate.py` | **new** `--c_bound` flag (must match checkpoint) |
| `scripts/reconstruct_designer.py` | `--c_bound`; prints axis clearance + warning |
| `scripts/verify_fixes.py` | audits the two new items |
| `docs/FINAL_SWEEP.md` | documents the failure, the measurements, the fix |

## Round 1 (the cleanup, if you have not pushed it yet)

| file | what changed |
|---|---|
| `scripts/train_multigpu.py` | **DELETE THIS FILE** (buggy; single-GPU is the path) |
| `nssr/networks.py` | removed `free_residual`; added `learn_heights` + `hhead` |
| `nssr/geometry.py` | removed `rho`; added `apply_cap_heights()` |
| `nssr/geometry_np.py` | removed `rho` from `zero_params_np` |
| `nssr/train.py` | removed `free_residual`; added `learn_heights`, `cap_weight` |
| `nssr/losses.py` | `chamfer_weighted()` (weights BOTH Chamfer directions) |
| `scripts/train_model.py` | `--no_learn_heights`, `--cap_weight`; dropped `--free_residual` |
| `scripts/reconstruct_designer.py` | `freeze_cap_params()`, `--tag`, descriptive filenames |
| `scripts/visualize_real.py` | `--pole_zoom`, `--pole_frac` |
| `scripts/ablation_caps.py` | **new file** (2x2 cap ablation) |
| `tests/cap_height_check.py` | **new file** |
| `run_all.sh` | cap ablation + freeze_caps renders + pole zoom wired in |

## Sanity check after updating

```bash
python scripts/verify_fixes.py     # must print "PASS -- this copy is current"
python tests/parity_check.py       # must still match your reference code
```
