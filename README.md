# Natural Primal-Dual Hybrid Gradient (NPDG) method

Collection of notebooks and Python scripts for Natural Primal-Dual Hybrid Gradient (NPDG) method across PDE and optimal transport examples presented in Section 5 of the manuscript arxiv.org/abs/2411.06278.

## Repository layout

- `Poisson/` (Sec. 5.1) Poisson equation example with training scripts implementing NPDG, PINN, DeepRitz, and WAN solvers.
- `Varcoeff/` (Sec. 5.2) Variable-coefficient elliptic example with training scripts implementing NPDG, PINN, DeepRitz, and WAN solvers.
- `Semi_linear/` (Sec. 5.3) Semi-linear equation example with training scripts implementing NPDG, PINN, and WAN solvers.
- `RD/NPDG_for_AllenCahn1D.ipynb` (Sec. 5.4) 1D Reaction Diffusion (Allen-Cahn) equation notebook.
- `RD2D/NPDG_for_AllenCahn2D.ipynb` (Sec. 5.4) 2D Reaction Diffusion (Allen-Cahn) equation notebook.
- `OT1D/OT1D.ipynb` (Sec. 5.5.1) 1D optimal transport notebook.
- `OT_Gaussian/OTGaussian.ipynb` (Sec. 5.5.2) Optimal transport between Gaussians notebook.
- `OTMixGaussian/NPDG_for_OT_Mixture_Gaussians.ipynb` (Sec. 5.5.3) Optimal transport between Gaussian mixtures notebook.

## Requirements

Typical environment:

- Python 3.8+
- `torch`, `numpy`, `scipy`, `matplotlib`
- `jupyter` for notebooks

## Quick start (scripts)

GPU is optional; the scripts will use CUDA if available.

Example NPDG runs:

```bash
python Semi_linear/run_npdhg.py
python Poisson/run_npdhg.py
python Varcoeff/run_npdhg.py
