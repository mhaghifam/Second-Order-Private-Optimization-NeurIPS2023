# Second-Order Private Optimization (NeurIPS 2023)


This repo accompanies the NeurIPS 2023 paper **“Faster Differentially Private Convex Optimization via Second-Order Methods.”**


---


For convex ERM (e.g., logistic regression / linear classifiers), standard DP first-order methods are slow because:
- privacy noise accumulates across many gradient steps,  
- so you need tiny steps + lots of iterations to stabilize training.

This code implements **Double-Noise DP Newton-style methods** that:
- leverage curvature to take larger, better-scaled steps,
- **reduce iteration count dramatically**, and
- match strong privacy guarantees.

Think: **faster DP baselines for privacy-sensitive analytics and model training.**

---

## What’s inside

### Algorithms
- **Double-Noise Mechanism (our method)** — 4 variants:
  - `DN-Hess-add`  : Hessian SOI + eigenvalue **add** regularization  
  - `DN-Hess-clip` : Hessian SOI + eigenvalue **clip** regularization  
  - `DN-UB-add`    : Quadratic upper-bound SOI + **add** regularization  
  - `DN-UB-clip`   : Quadratic upper-bound SOI + **clip** regularization  

- **Baselines**
  - `DPGD` : DP Gradient Descent  
  - `private-newton` : DP Damped Newton baseline  

### Datasets
Supported via `dataset_loader.py`:
- `a1a_dataset`
- `protein_dataset`
- `fmnist_dataset`
- `synthetic_dataset`

---

## Setup

### Requirements
- Python 3.8+
- NumPy
- SciPy

Install deps:
```bash
pip install numpy scipy
