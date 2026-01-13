# NPDG Code Collection

Collection of notebooks and Python scripts for neural primal-dual methods (NPDG/NPDHG) and baselines (PINN, DeepRitz, WAN) across PDE and optimal transport examples.

## Repository layout

- `HJB/` 5D Hamilton-Jacobi-Bellman example with training scripts and plotting utilities.
- `Poisson/` Poisson equation example with NPDHG, PINN, DeepRitz, and WAN solvers.
- `Varcoeff/` Variable-coefficient elliptic example with the same solver families.
- `RD/` Allen-Cahn 1D reaction-diffusion notebook.
- `RD2D/` Allen-Cahn 2D reaction-diffusion notebook.
- `OT1D/` 1D optimal transport notebook.
- `OT_Gaussian/` Gaussian optimal transport notebook.
- `OTMixGaussian/` Optimal transport between Gaussian mixtures notebook.

## Requirements

Typical environment:

- Python 3.8+
- `torch`, `numpy`, `scipy`, `matplotlib`
- `jupyter` for notebooks

GPU is optional; the scripts will use CUDA if available.

## Quick start (scripts)

Example NPDHG runs:

```bash
python HJB/run_npdhg.py
python Poisson/run_npdhg.py
python Varcoeff/run_npdhg.py
```

Baselines:

```bash
python HJB/run_pinn.py
python HJB/run_wan.py
python Poisson/run_deepritz.py
python Poisson/run_pinn.py
python Poisson/run_wan.py
python Varcoeff/run_deepritz.py
python Varcoeff/run_pinn.py
python Varcoeff/run_wan.py
```

Outputs are written to subfolders such as `NPDHG_experiments/` or `wan_experiments/` under each problem directory.

## Notebooks

Open any of the `.ipynb` files under `RD/`, `RD2D/`, `OT1D/`, `OT_Gaussian/`, or `OTMixGaussian/` in Jupyter. Some notebooks include hard-coded Colab paths (e.g., `/content/...`); update those paths to your local checkout before running.
