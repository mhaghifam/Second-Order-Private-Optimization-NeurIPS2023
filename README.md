# Second-Order Private Optimization (NeurIPS 2023)


[![NeurIPS 2023](https://img.shields.io/badge/NeurIPS-2023-blue.svg)]([https://neurips.cc/virtual/2024/poster/94421](https://proceedings.neurips.cc/paper_files/paper/2023/file/fb1d9c3fc2161e12aa71cdcab74b9d2c-Paper-Conference.pdf))
[![arXiv](https://img.shields.io/badge/arXiv-2305.13209-b31b1b.svg)]([https://arxiv.org/abs/2406.07407](https://arxiv.org/abs/2305.13209))


Official implementation of **"Private Geometric Median"** presented at NeurIPS 2024.

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


## Contact

For questions and feedback:
- **Mahdi Haghifam** - [haghifam.mahdi@gmail.com](mailto:haghifam.mahdi@gmail.com)


## Setup

### Requirements
- Python 3.8+
- NumPy
- SciPy

Install deps:
```bash
pip install numpy scipy



