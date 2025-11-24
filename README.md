Faster Differentially Private Convex Optimization via Second-Order Methods

This repository contains the official implementation of the algorithms and experiments presented in the NeurIPS 2023 paper "Faster Differentially Private Convex Optimization via Second-Order Methods".



🚀 Key Algorithms

Practical DP Newton with Double Noise (Algorithm 3)

A practical algorithm designed for logistic regression and Generalized Linear Models (GLMs). It privatizes the update by injecting noise at two stages:

Gradient Privatization: Adding noise to the gradient.

Direction Privatization: Adding noise to the Newton direction after scaling by the Hessian.

To ensure stability and privacy, the algorithm modifies the Hessian eigenvalues using two strategies:

Hessian Clipping (clip): Replaces small eigenvalues $\lambda_i$ with $\max\{\lambda_i, \lambda_0\}$.

Hessian Adding (add): Adds a constant shift $\lambda_i + \lambda_0$ (Regularization).



@inproceedings{ganesh2023faster,
  title={Faster Differentially Private Convex Optimization via Second-Order Methods},
  author={Ganesh, Arun and Haghifam, Mahdi and Steinke, Thomas and Thakurta, Abhradeep},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}


📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
