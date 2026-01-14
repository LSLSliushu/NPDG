# Natural Primal-Dual Hybrid Gradient (NPDG) method

Collection of notebooks and Python scripts for the Natural Primal-Dual Hybrid Gradient (NPDG) method across PDE and optimal transport examples presented in Section 5 of the manuscript:

> **A Natural Primal-Dual Hybrid Gradient Method for Adversarial Neural Network Training on Solving Partial Differential Equations**
> *Shu Liu, Stanley Osher, Wuchen Li*  
> *[arXiv:2411.06278](https://arxiv.org/abs/2411.06278)*

---

## Repository Layout

| Category | Path | Section | Description |
|:--|:--|:--|:--|
| Training Scripts | `Poisson/` | Sec. 5.1 | Poisson equation; NPDG, PINN, DeepRitz, and WAN solvers |
| Training Scripts | `Varcoeff/` | Sec. 5.2 | Variable-coefficient equation; NPDG, PINN, DeepRitz, and WAN solvers |
| Training Scripts | `Semi_linear/` | Sec. 5.3 | Semi-linear equation; NPDG, PINN, and WAN solvers |
| Notebooks | `RD/NPDG_for_AllenCahn1D.ipynb` | Sec. 5.4 | 1D Reaction–Diffusion (Allen–Cahn) equation |
| Notebooks | `RD2D/NPDG_for_AllenCahn2D.ipynb` | Sec. 5.4 | 2D Reaction–Diffusion (Allen–Cahn) equation |
| Notebooks | `OT1D/OT1D.ipynb` | Sec. 5.5.1 | 1D Optimal Transport |
| Notebooks | `OT_Gaussian/OTGaussian.ipynb` | Sec. 5.5.2 | Optimal Transport between Gaussian distributions |
| Notebooks | `OTMixGaussian/NPDG_for_OT_Mixture_Gaussians.ipynb` | Sec. 5.5.3 | Optimal Transport from Gaussian to Gaussian mixtures |

---

## Requirements

Typical environment:

- Python 3.8+
- `torch`, `numpy`, `scipy`, `matplotlib`
- `jupyter` for notebooks

---

## Scripts

:point_right: GPU is optional; the scripts will use CUDA if available.

Example NPDG runs:

```bash
python Poisson/run_npdhg.py
python Varcoeff/run_npdhg.py
python Semi_linear/run_npdhg.py
```

Baselines:

```bash
python Poisson/run_deepritz.py
python Poisson/run_pinn.py
python Poisson/run_wan.py
python Varcoeff/run_deepritz.py
python Varcoeff/run_pinn.py
python Varcoeff/run_wan.py
python Semi_linear/run_pinn.py
python Semi_linear/run_wan.py
```

Outputs are written to subfolders such as `NPDHG_experiments/` or `wan_experiments/` under each problem directory.

💡 Users can modify the problem dimension in `config.py`, the PDE parameters in `pde.py`, and the hyperparameters for each tested algorithm in `run_[name_of_method].py`.

---

## Notebooks

:point_right: We recommend running the notebooks in Google Colab with GPU acceleration enabled.

Open any of the `.ipynb` files under `RD/`, `RD2D/`, `OT1D/`, `OT_Gaussian/`, or `OTMixGaussian/` in Google Colab or Jupyter. Some notebooks include hard-coded Colab paths (e.g., `/content/...`); If you are not using Colab, please update those paths to your local checkout before running. Execute the cells sequentially to reproduce the results.

💡 Users can modify the problem dimension and the hyperparameters of each tested algorithm in the corresponding cells in each notebook.

---

**Contact:** sl25bn@fsu.edu / sliu11@fsu.edu

---
