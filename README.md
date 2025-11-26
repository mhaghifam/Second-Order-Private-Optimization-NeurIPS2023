# Second-Order Private Optimization (NeurIPS 2023)

This repo contains the code for the numerical results presented in the following paper. 

**"Faster Differentially Private Convex Optimization via Second-Order Methods"** <br>
**Published at NeurIPS'23**<br>
[[https://arxiv.org/abs/1911.02151](https://arxiv.org/abs/2305.13209)] <br>
by Arun Ganesh, Mahdi Haghifam, Thomas Steinke, Abhradeep Thakurta


---

This code implements **Double-Noise DP Newton-style methods** that:
- leverage curvature to take larger, better-scaled steps,
- **reduce iteration count dramatically**,

![My Project Screenshot](./intro-plot.jpg)

---

## What’s inside

### Algorithms
- **Double-Noise Mechanism (our method)** — 4 variants:
  - `DN-Hess-add`  : Hessian SOI + eigenvalue **add** regularization  
  - `DN-Hess-clip` : Hessian SOI + eigenvalue **clip** regularization  
  - `DN-UB-add`    : Quadratic upper-bound SOI + **add** regularization  
  - `DN-UB-clip`   : Quadratic upper-bound SOI + **clip** regularization  

- **Baselines**
  - `DPGD` : DP Gradient Descent  from [https://arxiv.org/abs/1405.7085](https://arxiv.org/abs/1405.7085)
  - `private-newton` : DP Damped Newton baseline  from [https://arxiv.org/abs/2103.11003](https://arxiv.org/abs/2103.11003) 

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



