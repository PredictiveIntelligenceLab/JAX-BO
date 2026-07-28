# jaxbo 0.2.0 revamp: scope

Status: DRAFT, pending Ricardo's approval. Refs ricardogr07/forja#13.

## 1. Why revamp

A fresh `pip install jaxbo` is broken on any modern stack. Both real consumers ship workarounds:

- **shelter-pulse** (`Dockerfile` lines 13 to 20) carries two shims: it writes a fake `pyDOE.py` module into site-packages (`from pydoe import *`) because jaxbo imports the dead `pyDOE` package name, and it runs `sed` over every installed `jaxbo/*.py` to rewrite `np.clip(x, a_min=...)` to positional form because the `a_min` keyword was removed in modern JAX (0.10+).
- **ecg-purkinje-npe** avoids PyPI entirely: it vendors a full editable copy at `packages/jax-bo` for its planned BO+ABC baseline.

The library also drags numpyro, scikit-learn, KDEpy, and pyDOE as hard install-time and import-time dependencies even for a consumer that only wants GP + EI.

The fix is a small 0.2.0 release that installs and imports clean. This is Forja OKR KR6. Arc: scope W1, implementation W2, release candidate W5.

## 2. What revamped jaxbo IS

A maintained, modern-JAX Bayesian optimization library with a small stable core: exact GP regression with a handful of standard kernels, the classic acquisition functions as pure jit-compiled functions, input priors, and L-BFGS training, nothing else. It installs clean on current jax/jaxlib with pinned, tested version ranges, has no dead dependencies, and documents its array-shape conventions. It is the BO engine for shelter-pulse and the ecg-purkinje-npe baseline, and a credible small library for anyone else.

## 3. Stable public API sketch (the contract)

Everything below is the supported surface for 0.2.x. Anything not shown is internal or dropped.

```python
import jax
import jax.numpy as jnp
from jaxbo.models import GP
from jaxbo import acquisitions, input_priors, kernels, optimizers

# Priors
prior = input_priors.uniform_prior(lb, ub)          # kept
prior = input_priors.gaussian_prior(mu, cov)        # kept

# GP: construct, train, predict (exact current signatures, unchanged)
gp = GP({"kernel": "Matern52", "input_prior": prior})   # kernels: RBF, Matern52, Matern32, Matern12, RatQuad
params = gp.train(batch, rng_key, num_restarts=3)       # batch = {"X": (N, D), "y": (N, 1)}
mu, std = gp.predict(X_star, params=params, batch=batch, bounds=bounds)

# Acquisitions: all 10 kept as pure single-point functions
# EI, EIC, LCB, LCBC, LW_LCB, LW_LCBC, US, LW_US, CLSF, LW_CLSF
a = acquisitions.EI(mu, std, best)

# NEW in 0.2.0: batched acquisition helper (the ergonomics fix).
# Today every caller hand-rolls a Python loop over candidates (shelter-pulse
# loops 256 Dirichlet candidates calling predict + EI one at a time) and has
# to know the undocumented mean[0, :] / std[0, :] row convention.
scores = acquisitions.score_candidates(
    gp, params, batch, bounds,
    X_cand,                      # (M, D) candidate points
    best,                        # incumbent
    acquisition=acquisitions.EI,
)                                # returns (M,) scores, vmap inside, jit once

# Optimizers: kept
x, f = optimizers.minimize_lbfgs(objective, x0, bnds=bnds)
x, f = optimizers.minimize_lbfgs_grad(objective_and_grad, x0, bnds=bnds)
```

Compatibility promise: the shelter-pulse call pattern above (`GP` dict options, `train(batch, rng_key, num_restarts)`, `predict(X_star, params=, batch=, bounds=)`, `EI(mu, std, best)`) works unchanged in 0.2.0. The batched helper is additive.

## 4. Keep / Fix / Drop per module

Consumer ground truth, verified by grep on 2026-07-28:

- **shelter-pulse** (`shelterpulse/optimize/jaxbo_optimizer.py`) imports exactly: `jaxbo.models.GP`, `jaxbo.acquisitions` (uses `EI`), `jaxbo.input_priors` (uses `uniform_prior`).
- **ecg-purkinje-npe** has **zero first-party imports of jaxbo**. It only declares the vendored `packages/jax-bo` as the optional `baseline` extra in `pyproject.toml`; the BO+ABC baseline code is not written yet. Nothing in that repo breaks under any drop below, and its planned baseline is the same GP + EI pattern the kept core serves.

Every Drop below is a proposal, **pending Ricardo**.

| Module | Verdict | Rationale | Consumer impact |
|---|---|---|---|
| `models/gp_model.py` (`GP`) | Keep + fix | The core class both consumers need | shelter-pulse imports it directly |
| `models/base_gpmodel.py` (`GPmodel`) | Fix | `GP` inherits from it. Replace `pyDOE` lhs, fix clip sites, strip the GMM/KDE weighted-sampling machinery (`fit_gmm`, `fit_kernel_density` path) that drags scikit-learn + KDEpy | None: shelter-pulse computes its own acquisition loop and never calls the weighted `compute_next_point` path |
| `kernels.py` (RBF, Matern52, Matern32, Matern12, RatQuad) | Keep | Pure jit functions, tested, zero cost | Matern52 used by shelter-pulse |
| `acquisitions.py` (10 functions) | Keep + fix | Pure jit functions, tested. Fix the one remaining `a_min` (EIC, line 45), document the `mean[0, :]` shape convention, add `score_candidates` | EI used by shelter-pulse |
| `input_priors.py` | Keep | Small, tested | `uniform_prior` used by shelter-pulse |
| `optimizers.py` | Keep | `minimize_lbfgs_grad` is the GP training engine | Indirect via `train` |
| `initializers.py` | Fix (trim) | Keep `random_init_GP`; drop the MultifidelityGP / GradientGP / SparseGP initializers with their models | None |
| `utils.py` | Fix (trim) | Keep the normalization helpers the GP path needs; fix its clip site; drop `fit_kernel_density` (KDEpy) with the weighted machinery | None |
| `test_functions.py` | Keep (drop candidate) | Zero extra deps, self-contained, covered by tests, useful for CI smoke tests and examples. Honest note: neither consumer imports it, so it is droppable, but it costs nothing and deleting it buys nothing. Trim the multifidelity test functions if the MF models drop | None |
| `mcmc_models.py` | **Drop, pending Ricardo** | Sole reason numpyro is a dependency, imported eagerly by `__init__.py` so every consumer pays for it. Also defines a second class named `GP` that shadows `models.GP`. No consumer imports it | None (verified) |
| `serializable.py` | **Drop, pending Ricardo** | `serializable_MF` / `deserializable_MF` only serialize multifidelity model params; falls with the MF family | None (verified) |
| `models/` multifidelity family: `MultifidelityGP`, `DeepMultifidelityGP`, `DeepMultifidelityGP_MultiOutputs`, `HeterogeneousMultifidelityGP`, `MultipleIndependentMFGP`, `MultipleIndependentHeterogeneousMFGP` | **Drop, pending Ricardo** | 6 of the 12 model classes; carry 10 of the 23 broken clip sites, 4 of the 6 dead `pyDOE` imports, and 2 of the 3 `sklearn.mixture` imports. Untested (no model tests exist at all). No consumer imports them | None (verified) |
| `models/` manifold family: `ManifoldGP`, `ManifoldGP_MultiOutputs` | **Drop, pending Ricardo** | Research variants (neural-net warped inputs), untested, no consumer | None (verified) |
| `models/gradient_gp.py` (`GradientGP`) | **Drop, pending Ricardo** | Untested, no consumer | None (verified) |
| `models/multiple_independent_output_gp_model.py` (`MultipleIndependentOutputsGP`) | **Drop, pending Ricardo** | Untested, no consumer, dead `pyDOE` import | None (verified) |
| `examples/` (12 notebooks + 2 scripts) + root `jaxbo_colab.ipynb` | **Drop, pending Ricardo** | All import `pyDOE` or exercise dropped models; none run on a modern install. Replace with one quickstart notebook built against 0.2.0 | None |

Net effect of the proposed drops: dependencies shrink from 8 to 4 (`numpy`, `scipy`, `jax`, `jaxlib`; numpyro, scikit-learn, KDEpy, pyDOE all go), and 20 of the 23 broken clip sites plus all 6 dead `pyDOE` imports disappear by deletion rather than patching.

## 5. Mechanical fixes (regardless of the drop list)

1. **pyDOE replacement.** scipy is already a dependency, so the surviving `lhs(dim, n)` call sites in `base_gpmodel.py` (lines 183, 189, 388) become one line:

   ```python
   from scipy.stats import qmc
   X = lb + (ub - lb) * qmc.LatinHypercube(d=dim, seed=seed).random(n)
   ```

   If we ever want scipy gone too, the numpy equivalent is five lines:

   ```python
   def lhs(dim, n, rng):
       """Latin hypercube sample in [0, 1]^dim, shape (n, dim)."""
       u = rng.uniform(size=(n, dim))
       perms = np.argsort(rng.uniform(size=(n, dim)), axis=0)
       return (perms + u) / n
   ```

   Recommendation: the scipy one-liner. Bonus over pyDOE: it is seedable, the old path was not.

2. **clip sweep.** 23 sites use the removed `a_min=` keyword (17 in `models/`, 4 in `mcmc_models.py`, 1 in `utils.py`, 1 in `acquisitions.py` EIC; `acquisitions.py` EI at line 25 was already migrated, so the codebase is inconsistent today). After the proposed drops only 3 sites remain (`gp_model.py:101`, `utils.py:367`, `acquisitions.py:45`); sweep whatever survives to positional `np.clip(x, 0.0, None)` and add a repo grep check so `a_min` never returns.

3. **Pin jax/jaxlib.** Unpinned jax is the root cause of the clip breakage. Pin to a tested range: `jax>=0.6,<0.11` and matching `jaxlib`, with the CI matrix exercising both the floor and the latest release. Widen the ceiling only when CI proves it.

4. **Honest requires-python.** Metadata says `>=3.6`; tox and CI only ever test 3.10 and 3.12, and modern jax itself requires 3.10+. Recommend `requires-python = ">=3.10"`: claiming less is untested fiction, and it costs no real users.

5. **Metadata and CI cleanup.** Remove dropped deps from `pyproject.toml`; remove the eager `mcmc_models` import (and other dropped modules) from `__init__.py`; keep tox py310/py312 + black + ruff as is; make `test.yml` run the jax floor/latest matrix; keep release-please driving the changelog (CHANGELOG is already conventional-commits based since 0.1.0).

6. **Tests for the core.** Today `tests/` covers acquisitions, kernels, optimizers, priors, test functions, utils, but **not one of the 12 model classes**. 0.2.0 adds a `GP` train/predict round-trip test (synthetic 1D and 4D objective, assert predictive mean recovers noiseless truth within tolerance) plus a `score_candidates` shape test. That is the regression net for everything above.

## 6. Release path: 0.2.0

1. Land the drop list + mechanical fixes + batched helper on `main` via PRs, release-please cuts `0.2.0` (minor bump: additive helper plus removals from an unstable 0.x surface, changelog gets an explicit "Removed" section listing every dropped module).
2. **Validation gate, shelter-pulse:** bump to `jaxbo==0.2.0`, delete both Dockerfile shims (the fake `pyDOE.py` writer and the `sed` clip rewrite), image builds clean, BO smoke run produces candidates via the GP+EI path (log line "real GP+EI optimization available").
3. **Validation gate, ecg-purkinje-npe:** replace `jaxbo = { path = "packages/jax-bo", editable = true }` with the PyPI `jaxbo>=0.2,<0.3` in the `baseline` extra, delete `packages/jax-bo`, `uv lock` and import check pass.
4. CI matrix on jaxbo itself: {py310, py312} x {jax floor, jax latest}, plus lint. Green matrix is the RC bar (W5).
5. Changelog discipline: conventional commits only, release-please owns versioning, every future removal or signature change gets a changelog line before merge.

## 7. Open decisions for Ricardo

1. **Drop list** (section 4): drop `mcmc_models`, `serializable`, the 6 multifidelity models, the 2 manifold models, `GradientGP`, `MultipleIndependentOutputsGP`, and the current `examples/`. Recommendation: **approve all**. Verified: no consumer imports any of them, and the deletion removes 4 dependencies and 20 broken sites for free. Anyone needing the research variants has the upstream PredictiveIntelligenceLab repo and our git history.
2. **Python floor**: recommend **3.10** (matches tox, CI, and modern jax reality; `>=3.6` is untested fiction).
3. **jax pin**: recommend **`jax>=0.6,<0.11`** with floor+latest in CI, ceiling raised only on green CI. Alternative is floor-only (`>=0.6`), but an unpinned ceiling is exactly what broke 0.1.x.
4. **Batched helper timing**: recommend **land `score_candidates` in 0.2.0**. It is ~15 lines around vmap, it is the ergonomics friction that motivated the revamp, and shipping it with the release lets shelter-pulse delete its 256-candidate Python loop in the same bump. Deferring it saves almost nothing.
5. **test_functions**: recommend **keep** (trimmed of multifidelity variants). Droppable in principle, but it is dependency-free, already tested, and useful for CI smoke tests and the quickstart.
6. **Weighted-acquisition machinery** (`fit_gmm` + `fit_kernel_density` inside the kept `GPmodel`): recommend **drop**, it is what forces scikit-learn and KDEpy on every install. The `LW_*` acquisition functions stay (they take precomputed weights as plain arguments).
7. **Examples**: recommend **delete and replace with one quickstart notebook** that runs top to bottom against 0.2.0 in CI-adjacent fashion (or is at least import-checked).

## 8. Proposed issue breakdown (becomes `orchestration/epics/jaxbo-revamp.json` after approval)

| # | Slice | Size |
|---|---|---|
| 1 | Trim the package: delete approved drop-list modules, update `__init__.py` and `models/__init__.py` exports, trim `initializers`/`utils`, strip GMM/KDE machinery from `GPmodel` | M |
| 2 | Replace `pyDOE` with `scipy.stats.qmc.LatinHypercube` at the surviving call sites, delete the dependency | S |
| 3 | Sweep remaining `clip(..., a_min=)` sites to positional form, add a grep guard to lint/CI | S |
| 4 | Packaging: pin `jax`/`jaxlib`, `requires-python >= 3.10`, prune dependency list, metadata polish | S |
| 5 | Add `acquisitions.score_candidates` (vmap batched helper) with shape-convention docstrings | M |
| 6 | Model tests: `GP` train/predict round-trip (1D + 4D), `score_candidates` shapes, EI sanity | M |
| 7 | CI: {py310, py312} x {jax floor, jax latest} matrix, keep lint, release-please for 0.2.0 with a Removed section | S |
| 8 | README + quickstart notebook rewritten against the 0.2.0 surface | S |
| 9 | Consumer validation: shelter-pulse deletes both Dockerfile shims and builds clean on 0.2.0; ecg-purkinje-npe swaps the vendored copy for the PyPI dep | M |

Total: 4 S + 3 M in jaxbo, 1 S (docs) and 1 M (cross-repo validation). Slices 1 to 4 are the W2 implementation core; 5 to 8 harden; 9 is the RC gate (W5).
