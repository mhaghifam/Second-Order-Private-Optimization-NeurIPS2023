Faster Differentially Private Convex Optimization via Second-Order Methods

This repository contains the official implementation of the algorithms and experiments presented in the NeurIPS 2023 paper "Faster Differentially Private Convex Optimization via Second-Order Methods".

[Paper Link] <!-- Replace with actual Arxiv/NeurIPS link if different -->

📝 Abstract

Differentially private (stochastic) gradient descent (DP-SGD) is the standard for private machine learning. However, without privacy constraints, second-order methods (like Newton's method) often converge significantly faster than first-order methods.

In this work, we investigate using second-order information (Hessian) to accelerate DP convex optimization. We introduce:

A Private Cubic Regularized Newton Method for strongly convex loss functions, which achieves optimal excess loss with quadratic convergence.

A Practical Second-Order DP Algorithm for unconstrained logistic regression that injects carefully designed noise into Newton's update rule.

Empirical results demonstrate that our method is 10-40x faster than DP-GD/DP-SGD on challenging datasets while consistently achieving better excess loss.

🚀 Key Algorithms

This repository implements the following algorithms discussed in the paper:

1. Private Cubic Regularized Newton (Algorithm 1)

A theoretical framework based on Nesterov and Polyak's method. It minimizes a cubic upper bound of the loss function at each step.

Target: Strongly convex loss functions.

Guarantee: Optimal excess loss and quadratic convergence.

2. Practical DP Newton with Double Noise (Algorithm 3)

A practical algorithm designed for logistic regression and Generalized Linear Models (GLMs). It privatizes the update by injecting noise at two stages:

Gradient Privatization: Adding noise to the gradient.

Direction Privatization: Adding noise to the Newton direction after scaling by the Hessian.

To ensure stability and privacy, the algorithm modifies the Hessian eigenvalues using two strategies:

Hessian Clipping (clip): Replaces small eigenvalues $\lambda_i$ with $\max\{\lambda_i, \lambda_0\}$.

Hessian Adding (add): Adds a constant shift $\lambda_i + \lambda_0$ (Regularization).

💻 Installation

Clone the repository and install the dependencies:

git clone [https://github.com/mhaghifam/Second-Order-Private-Optimization-NeurIPS2023.git](https://github.com/mhaghifam/Second-Order-Private-Optimization-NeurIPS2023.git)
cd Second-Order-Private-Optimization-NeurIPS2023
pip install -r requirements.txt


Note: The code requires standard Python scientific libraries such as numpy, scipy, and matplotlib.

📊 Usage

Running Experiments

To reproduce the experiments from the paper (e.g., Logistic Regression on the Covertype dataset), use the main execution script.

Example command (adjust arguments based on actual code structure):

python main.py --dataset covertype --algorithm newton_double_noise --epsilon 1.0 --method hessian_clip


Parameters

--dataset: Choices include ala, adult, covertype, synthetic, fashion-mnist, protein.

--epsilon: Privacy budget ($\epsilon$).

--method: hessian_clip or hessian_add.

📂 Datasets

The experiments in the paper are conducted on the following datasets:

Synthetic: Generated data on the unit sphere.

a1a / Adult: Standard UCI repository datasets.

Covertype: Forest cover type prediction (UCI).

Fashion-MNIST: Image classification benchmarks.

Protein: Biology dataset.

📈 Results

Our second-order methods achieve superior privacy-utility trade-offs compared to baselines like DP-GD and Damped Newton methods.

Dataset

Speedup vs DP-GD

a1a

~4-5x

Adult

~12-20x

Covertype

~20-35x

Synthetic

~3-5x

📚 Citation

If you use this code or find the paper useful in your research, please cite:

@inproceedings{ganesh2023faster,
  title={Faster Differentially Private Convex Optimization via Second-Order Methods},
  author={Ganesh, Arun and Haghifam, Mahdi and Steinke, Thomas and Thakurta, Abhradeep},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023}
}


📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
