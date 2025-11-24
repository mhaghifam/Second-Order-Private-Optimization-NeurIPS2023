
"""Optimization algorithms for differentially private machine learning.

This module implements various optimization algorithms including:
- Differentially private gradient descent (DP-GD)
- Differentially private stochastic gradient descent (DP-SGD)
- Private Newton method
- Double noise mechanism for second-order optimization
"""

import time
from typing import Dict, List, Tuple, Optional, Callable, Literal
import numpy as np

from my_logistic_regression import MyLogisticRegression

# Optimization constants
DEFAULT_MIN_EIGENVALUE = 1e-5
DEFAULT_REGULARIZATION = 1e-9
NEWTON_ITERATIONS = 8
LINE_SEARCH_MIN_STEP = 1e-6

# Line search parameters
LINE_SEARCH_ALPHA = 0.4  # Sufficient decrease parameter
LINE_SEARCH_BETA = 0.95  # Step size reduction factor
INITIAL_STEP_SIZE = 100.0

# Algorithm-specific constants
GD_LEARNING_RATE_INVERSE = 0.25  # Based on smoothness
SGD_LEARNING_RATE = 4.0  # Learning rate for SGD
HESSIAN_NOISE_SCALE = 0.25  # Scale factor for Hessian noise
GRADIENT_NOISE_SCALE = 1.0  # Scale factor for gradient noise
SMOOTHNESS_PARAMETER = 0.25

# Stability constants
NORM_EXPLOSION_FACTOR = 100.0  # Multiplier for detecting parameter explosion


class AlgorithmComparison:
    """Compare multiple iterative optimization algorithms on the same problem.
    
    This class runs various optimization algorithms on a logistic regression
    problem and collects metrics like loss values, accuracy, gradient norms,
    and wall-clock time for comparison.
    
    Attributes:
        optimal_weights: Known optimal solution without privacy constraints
        logistic_model: MyLogisticRegression instance defining the problem
        n_iterations: Number of iterations to run each algorithm
        hyperparameters: Dictionary of algorithm hyperparameters
        wall_times: List of wall-clock times for each algorithm
        weight_trajectories: List of weight vectors throughout optimization
        algorithm_names: Names of the algorithms being compared
    """
    
    def __init__(
        self, 
        logistic_model: MyLogisticRegression, 
        optimal_weights: np.ndarray, 
        hyperparameters: Dict
    ):
        """Initialize algorithm comparison framework.
        
        Args:
            logistic_model: Logistic regression problem instance
            optimal_weights: Known optimal weights (without privacy)
            hyperparameters: Dictionary containing:
                - num_iteration: Number of iterations to run
                - total: Total privacy budget
                - grad_frac: Fraction of privacy budget for gradient
                - Other algorithm-specific parameters
        """
        self.optimal_weights = optimal_weights
        self.logistic_model = logistic_model
        self.n_iterations = hyperparameters["num_iteration"]
        self.hyperparameters = hyperparameters
        self.wall_times: List[List[float]] = []
        self.weight_trajectories: List[List[np.ndarray]] = []
        self.algorithm_names: List[str] = []
    
    def add_algorithm(
        self, 
        update_function: Callable, 
        algorithm_name: str
    ):
        """Run an optimization algorithm and store its results.
        
        Executes the given update function for the specified number of iterations,
        tracking wall-clock time and parameter values. Includes stability checks
        to prevent numerical explosion.
        
        Args:
            update_function: Function that takes (weights, model, hyperparams) 
                           and returns updated weights
            algorithm_name: Name identifier for this algorithm
        """
        n_features = self.logistic_model.dim
        
        # Random initialization on unit sphere
        initial_weights_unnormalized = np.random.multivariate_normal(
            np.zeros(n_features), 
            np.eye(n_features)
        )
        initial_weights = initial_weights_unnormalized / np.linalg.norm(
            initial_weights_unnormalized
        )
        
        # Set cutoff for detecting numerical instability
        max_allowed_norm = NORM_EXPLOSION_FACTOR * (
            np.linalg.norm(self.optimal_weights) + 
            np.linalg.norm(initial_weights) + 1.0
        )
        
        # Initialize trajectory tracking
        current_weights = initial_weights
        trajectory = [current_weights]
        start_time = time.time()
        time_points = [0.0]
        
        # Run optimization iterations
        for iteration in range(self.n_iterations):
            # Update weights
            current_weights = update_function(
                current_weights, 
                self.logistic_model, 
                self.hyperparameters
            )
            
            # Check for numerical explosion and reset if needed
            if np.linalg.norm(current_weights) > max_allowed_norm:
                current_weights = initial_weights
                print(f"Warning: Numerical explosion detected in {algorithm_name}, resetting weights")
            
            # Track progress
            trajectory.append(current_weights)
            time_points.append(time.time() - start_time)
        
        # Store results
        self.wall_times.append(time_points)
        self.weight_trajectories.append(trajectory)
        self.algorithm_names.append(algorithm_name)
    
    def get_wall_clock_times(self) -> Dict[str, List[List[float]]]:
        """Get wall-clock times for all algorithms.
        
        Returns:
            Dictionary mapping algorithm names to lists of time points
        """
        return {
            name: [times] 
            for times, name in zip(self.wall_times, self.algorithm_names)
        }
    
    def get_loss_values(self) -> Dict[str, List[List[float]]]:
        """Compute excess loss (above optimal) for each algorithm.
        
        Returns:
            Dictionary mapping algorithm names to lists of excess loss values
        """
        optimal_loss = self.logistic_model.loss_wor(self.optimal_weights)
        
        loss_values = {}
        for trajectory, name in zip(self.weight_trajectories, self.algorithm_names):
            excess_losses = [
                self.logistic_model.loss_wor(weights) - optimal_loss 
                for weights in trajectory
            ]
            loss_values[name] = [excess_losses]
        
        return loss_values
    
    def get_accuracy_values(self) -> Dict[str, List[List[float]]]:
        """Compute classification accuracy throughout optimization.
        
        Returns:
            Dictionary mapping algorithm names to lists of accuracy values
        """
        accuracy_values = {}
        for trajectory, name in zip(self.weight_trajectories, self.algorithm_names):
            accuracies = [
                self.logistic_model.accuracy(weights) 
                for weights in trajectory
            ]
            accuracy_values[name] = [accuracies]
        
        return accuracy_values
    
    def get_optimal_accuracy(self) -> float:
        """Get accuracy of the optimal model (without privacy).
        
        Returns:
            Classification accuracy of optimal weights
        """
        return self.logistic_model.accuracy(self.optimal_weights)
    
    def get_gradient_norms(self) -> Dict[str, List[List[float]]]:
        """Compute gradient norms throughout optimization.
        
        Returns:
            Dictionary mapping algorithm names to lists of gradient norm values
        """
        gradient_norms = {}
        for trajectory, name in zip(self.weight_trajectories, self.algorithm_names):
            norms = [
                np.linalg.norm(self.logistic_model.grad_wor(weights))
                for weights in trajectory
            ]
            gradient_norms[name] = [norms]
        
        return gradient_norms
    
 
    wall_clock_alg = get_wall_clock_times
    loss_vals = get_loss_values
    accuracy_vals = get_accuracy_values
    accuracy_np = get_optimal_accuracy
    gradnorm_vals = get_gradient_norms


def differentially_private_newton(
    current_weights: np.ndarray,
    logistic_model: MyLogisticRegression,
    hyperparameters: Dict
) -> np.ndarray:
    """Private Newton method update step from [ABL21].
    
    Implements one iteration of the differentially private Newton method
    with gradient and Hessian noise calibrated to the privacy budget.
    
    Args:
        current_weights: Current parameter vector
        logistic_model: Logistic regression model instance
        hyperparameters: Dictionary containing:
            - total: Total privacy budget
            - grad_frac: Fraction allocated to gradient
            - num_iteration: Total number of iterations
            
    Returns:
        Updated weight vector after one Newton step
    """
    # Extract privacy parameters
    total_budget = hyperparameters["total"]
    gradient_fraction = hyperparameters["grad_frac"]
    n_iterations = hyperparameters["num_iteration"]
    n_samples = logistic_model.num_samples
    n_features = logistic_model.dim
    
    # Allocate privacy budget between gradient and Hessian
    gradient_privacy = gradient_fraction * total_budget / n_iterations
    hessian_privacy = (1 - gradient_fraction) * total_budget / n_iterations
    
    # Add noise to Hessian
    hessian = logistic_model.hess(current_weights)
    hessian_noise_std = (HESSIAN_NOISE_SCALE / n_samples) * np.sqrt(0.5 / hessian_privacy)
    hessian_noise = np.random.normal(
        scale=hessian_noise_std,
        size=(n_features, n_features)
    )
    # Make noise symmetric
    hessian_noise = (hessian_noise + hessian_noise.T) / 2
    noisy_hessian = eigenvalue_clipping(hessian + hessian_noise)
    
    # Add noise to gradient
    gradient = logistic_model.grad(current_weights)
    gradient_noise_std = (GRADIENT_NOISE_SCALE / n_samples) * np.sqrt(0.5 / gradient_privacy)
    gradient_noise = np.random.normal(scale=gradient_noise_std, size=n_features)
    noisy_gradient = gradient + gradient_noise
    
    # Compute Newton direction
    noisy_direction = np.linalg.solve(noisy_hessian, noisy_gradient)
    
    # Adaptive step size based on clean Newton step norm
    clean_direction = np.linalg.solve(hessian, gradient)
    clean_step_norm = np.linalg.norm(clean_direction)
    adaptive_step_size = min(np.log(1 + clean_step_norm) / clean_step_norm, 1.0)
    
    return current_weights - adaptive_step_size * noisy_direction


def eigenvalue_clipping(
    symmetric_matrix: np.ndarray, 
    min_eigenvalue: float = DEFAULT_MIN_EIGENVALUE
) -> np.ndarray:
    """Clip eigenvalues of symmetric matrix to ensure positive definiteness.
    
    Projects the matrix onto the cone of positive definite matrices by
    clipping all eigenvalues below a threshold.
    
    Args:
        symmetric_matrix: Symmetric matrix to clip
        min_eigenvalue: Minimum allowed eigenvalue (default 1e-5)
        
    Returns:
        Modified matrix with all eigenvalues >= min_eigenvalue
    """
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_matrix)
    clipped_eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    clipped_matrix = np.dot(
        eigenvectors * clipped_eigenvalues, 
        eigenvectors.T
    )
    return clipped_matrix


def differentially_private_gradient_descent(
    current_weights: np.ndarray,
    logistic_model: MyLogisticRegression,
    hyperparameters: Dict
) -> np.ndarray:
    """Differentially private gradient descent (DP-GD) update step.
    
    Implements one iteration of gradient descent with Gaussian noise
    calibrated to ensure differential privacy.
    
    Args:
        current_weights: Current parameter vector
        logistic_model: Logistic regression model instance
        hyperparameters: Dictionary containing:
            - total: Total privacy budget
            - num_iteration: Total number of iterations
            
    Returns:
        Updated weight vector after one GD step
    """
    n_iterations = hyperparameters["num_iteration"]
    n_samples = logistic_model.num_samples
    n_features = logistic_model.dim
    
    # Privacy budget per iteration
    iteration_privacy = hyperparameters["total"] / n_iterations
    
    # Compute gradient
    gradient = logistic_model.grad_wor(current_weights)
    
    # Add calibrated noise for privacy
    sensitivity = 1.0 / (n_samples * GD_LEARNING_RATE_INVERSE)
    noise_std = sensitivity / np.sqrt(2 * iteration_privacy)
    noise = np.random.normal(scale=noise_std, size=n_features)
    
    # Gradient descent update
    return current_weights - gradient / GD_LEARNING_RATE_INVERSE + noise


def differentially_private_sgd(
    current_weights: np.ndarray,
    logistic_model: MyLogisticRegression,
    hyperparameters: Dict
) -> np.ndarray:
    """Differentially private stochastic gradient descent (DP-SGD) update.
    
    Implements one iteration of mini-batch SGD with per-sample gradient
    clipping and Gaussian noise for privacy.
    
    Args:
        current_weights: Current parameter vector
        logistic_model: Logistic regression model instance
        hyperparameters: Dictionary containing:
            - batch_size: Mini-batch size
            - noise_multiplier: Noise scale parameter
            
    Returns:
        Updated weight vector after one SGD step
    """
    batch_size = hyperparameters["batch_size"]
    noise_multiplier = hyperparameters["noise_multiplier"]
    n_samples = logistic_model.num_samples
    n_features = logistic_model.dim
    
    # Sample mini-batch using Poisson sampling
    sampling_probability = batch_size / n_samples
    sample_indicators = np.random.binomial(
        n=1, 
        p=sampling_probability, 
        size=n_samples
    )
    batch_indices = np.where(sample_indicators == 1)[0]
    actual_batch_size = len(batch_indices)
    
    # Compute mini-batch gradient
    minibatch_gradient = logistic_model.grad_wor(current_weights, batch_indices)
    
    # Add noise for privacy
    noise = np.random.normal(scale=noise_multiplier, size=n_features)
    
    # Scale gradient and noise appropriately
    scaled_gradient = (actual_batch_size / batch_size) * minibatch_gradient
    scaled_noise = noise / batch_size
    
    # SGD update
    return current_weights - SGD_LEARNING_RATE * (scaled_gradient + scaled_noise)


def dp_gd_with_line_search(
    current_weights: np.ndarray,
    logistic_model: MyLogisticRegression,
    hyperparameters: Dict
) -> np.ndarray:
    """DP-GD with backtracking line search (NOT PRIVATE - baseline only).
    
    WARNING: This method uses backtracking line search which leaks information
    about the data. It should only be used as a non-private baseline for
    comparison purposes.
    
    Args:
        current_weights: Current parameter vector
        logistic_model: Logistic regression model instance
        hyperparameters: Dictionary containing privacy parameters
        
    Returns:
        Updated weight vector
    """
    n_iterations = hyperparameters["num_iteration"]
    n_samples = logistic_model.num_samples
    n_features = logistic_model.dim
    
    # Privacy budget per iteration
    iteration_privacy = hyperparameters["total"] / n_iterations
    
    # Add noise to gradient
    gradient = logistic_model.grad(current_weights)
    noise_scale = (1.0 / n_samples) * np.sqrt(0.5 / iteration_privacy)
    gradient_noise = np.random.normal(scale=noise_scale, size=n_features)
    noisy_gradient = gradient + gradient_noise
    
    # Find optimal step size (THIS BREAKS PRIVACY)
    optimal_step = backtracking_line_search(
        logistic_model, 
        noisy_gradient, 
        current_weights
    )
    
    return current_weights - optimal_step * noisy_gradient


def backtracking_line_search(
    logistic_model: MyLogisticRegression,
    search_direction: np.ndarray,
    current_weights: np.ndarray,
    alpha: float = LINE_SEARCH_ALPHA,
    beta: float = LINE_SEARCH_BETA
) -> float:
    """Find step size using backtracking line search with Armijo condition.
    
    Args:
        logistic_model: Logistic regression model
        search_direction: Direction vector for line search
        current_weights: Current parameter vector
        alpha: Sufficient decrease parameter (default 0.4)
        beta: Step size reduction factor (default 0.95)
        
    Returns:
        Near-optimal step size
    """
    step_size = INITIAL_STEP_SIZE
    initial_loss = logistic_model.loss(current_weights)
    gradient_dot_direction = np.dot(
        search_direction, 
        logistic_model.grad(current_weights)
    )
    
    # Backtrack until Armijo condition satisfied
    while (logistic_model.loss(current_weights - step_size * search_direction) 
           >= initial_loss - step_size * alpha * gradient_dot_direction):
        step_size *= beta
        if step_size < LINE_SEARCH_MIN_STEP:
            break
    
    return step_size


def newton_method_non_private(
    dataset: Tuple[np.ndarray, np.ndarray],
    initial_weights: np.ndarray,
    add_bias: bool = True
) -> np.ndarray:
    """Standard Newton method with line search (non-private).
    
    Args:
        dataset: Tuple of (features, labels)
        initial_weights: Starting point for optimization
        add_bias: Whether to add bias term to features
        
    Returns:
        Optimized weight vector
    """
    features, labels = dataset
    
    # Add bias column if requested
    if add_bias:
        n_samples = features.shape[0]
        bias_column = np.ones((n_samples, 1))
        features = np.hstack((bias_column, features))
    
    # Initialize model
    model = MyLogisticRegression(features, labels, reg=DEFAULT_REGULARIZATION)
    
    current_weights = initial_weights
    for _ in range(NEWTON_ITERATIONS):
        # Compute Newton direction
        hessian = model.hess(current_weights)
        gradient = model.grad_wor(current_weights)
        search_direction = np.linalg.solve(hessian, gradient)
        
        # Line search for step size
        step_size = backtracking_line_search(model, search_direction, current_weights)
        
        # Update weights (NOTE: Fixed typo from original - was 'dir' instead of 'search_direction')
        current_weights = current_weights - step_size * search_direction
    
    # Return best weights
    if model.loss_wor(current_weights) < model.loss_wor(initial_weights):
        return current_weights
    else:
        return initial_weights


class DoubleNoiseMechanism:
    """Double noise mechanism for differentially private second-order optimization.
    
    This class implements a novel approach to private optimization that adds
    noise at two stages: to the gradient and to the Newton direction computation.
    It supports both full-batch and stochastic variants.
    
    Attributes:
        regularization_type: Type of eigenvalue regularization ('add' or 'clip')
        curvature_type: Type of second-order information ('hessian' or 'upperbound')
        hessian_function: Function to compute Hessian or its upper bound
    """
    
    def __init__(
        self,
        logistic_model: MyLogisticRegression,
        regularization_type: Literal["add", "clip"] = "add",
        curvature_type: Literal["hessian", "ub"] = "hessian"
    ):
        """Initialize double noise mechanism.
        
        Args:
            logistic_model: Logistic regression model instance
            regularization_type: How to handle minimum eigenvalue:
                - 'add': Add regularization to diagonal
                - 'clip': Clip small eigenvalues
            curvature_type: Second-order information type:
                - 'hessian': Use exact Hessian
                - 'ub': Use quadratic upper bound
        """
        self.regularization_type = regularization_type
        self.curvature_type = curvature_type
        
        # Select appropriate Hessian computation
        if self.curvature_type == "hessian":
            self.hessian_function = logistic_model.hess_wor
        elif self.curvature_type == "ub":
            self.hessian_function = logistic_model.upperbound_wor
        else:
            raise ValueError(f"Unknown curvature type: {curvature_type}")
    
    def update_full_batch(
        self,
        current_weights: np.ndarray,
        logistic_model: MyLogisticRegression,
        hyperparameters: Dict
    ) -> np.ndarray:
        """Full-batch update with double noise mechanism.
        
        Args:
            current_weights: Current parameter vector
            logistic_model: Logistic regression model
            hyperparameters: Algorithm hyperparameters
            
        Returns:
            Updated weight vector
        """
        # Compute noisy gradient
        noisy_gradient = self._compute_noisy_gradient(
            current_weights, 
            logistic_model, 
            hyperparameters, 
            use_batch=False
        )
        
        # Compute noisy Newton direction
        next_weights = self._compute_noisy_direction(
            current_weights, 
            logistic_model, 
            hyperparameters, 
            noisy_gradient
        )
        
        return next_weights
    
    def update_stochastic(
        self,
        current_weights: np.ndarray,
        logistic_model: MyLogisticRegression,
        hyperparameters: Dict
    ) -> np.ndarray:
        """Stochastic update with double noise mechanism.
        
        Args:
            current_weights: Current parameter vector
            logistic_model: Logistic regression model
            hyperparameters: Algorithm hyperparameters
            
        Returns:
            Updated weight vector
        """
        # Compute noisy gradient with mini-batch
        noisy_gradient = self._compute_noisy_gradient(
            current_weights, 
            logistic_model, 
            hyperparameters, 
            use_batch=True
        )
        
        # Compute noisy Newton direction with mini-batch
        next_weights = self._compute_noisy_direction_stochastic(
            current_weights, 
            logistic_model, 
            hyperparameters, 
            noisy_gradient
        )
        
        return next_weights
    
    def _compute_noisy_gradient(
        self,
        current_weights: np.ndarray,
        logistic_model: MyLogisticRegression,
        hyperparameters: Dict,
        use_batch: bool = False
    ) -> np.ndarray:
        """Compute gradient with calibrated noise for privacy.
        
        Args:
            current_weights: Current parameter vector
            logistic_model: Logistic regression model
            hyperparameters: Algorithm hyperparameters
            use_batch: Whether to use mini-batch sampling
            
        Returns:
            Noisy gradient vector
        """
        n_samples = logistic_model.num_samples
        n_features = logistic_model.dim
        
        if not use_batch:
            # Full-batch gradient with noise
            gradient_privacy = (
                hyperparameters["grad_frac"] * 
                hyperparameters["total"] / 
                hyperparameters["num_iteration"]
            )
            noise_scale = (1.0 / n_samples) * np.sqrt(0.5 / gradient_privacy)
            noise = np.random.normal(scale=noise_scale, size=n_features)
            gradient = logistic_model.grad(current_weights)
            noisy_gradient = gradient + noise
            
        else:
            # Mini-batch gradient with noise
            noise_std = hyperparameters["noise_multiplier_grad"]
            batch_probability = hyperparameters["batchsize_grad"] / n_samples
            
            # Sample mini-batch
            sample_indicators = np.random.binomial(
                n=1, 
                p=batch_probability, 
                size=n_samples
            )
            batch_indices = np.where(sample_indicators == 1)[0]
            actual_batch_size = len(batch_indices)
            
            # Compute gradient on mini-batch
            minibatch_gradient = logistic_model.grad_wor(
                current_weights, 
                batch_indices
            )
            
            # Add noise and scale appropriately
            noise = np.random.normal(scale=noise_std, size=n_features)
            noisy_gradient = (
                (actual_batch_size / (n_samples * batch_probability)) * minibatch_gradient + 
                noise / (n_samples * batch_probability)
            )
        
        return noisy_gradient
    
    def _compute_noisy_direction(
        self,
        current_weights: np.ndarray,
        logistic_model: MyLogisticRegression,
        hyperparameters: Dict,
        noisy_gradient: np.ndarray
    ) -> np.ndarray:
        """Compute Newton direction with noise for privacy (full-batch).
        
        Args:
            current_weights: Current parameter vector
            logistic_model: Logistic regression model
            hyperparameters: Algorithm hyperparameters
            noisy_gradient: Previously computed noisy gradient
            
        Returns:
            Next weight vector
        """
        n_samples = logistic_model.num_samples
        n_features = logistic_model.dim
        
        # Extract hyperparameters
        total_budget = hyperparameters["total"]
        gradient_fraction = hyperparameters["grad_frac"]
        trace_fraction = hyperparameters["trace_frac"]
        trace_coefficient = hyperparameters["trace_coeff"]
        n_iterations = hyperparameters["num_iteration"]
        
        # Privacy budget for Hessian
        hessian_privacy = (1 - gradient_fraction) * total_budget / n_iterations
        
        # Compute Hessian
        hessian = self.hessian_function(current_weights)
        
        # Compute noisy trace for eigenvalue regularization
        trace_noise_scale = (
            (HESSIAN_NOISE_SCALE / n_samples) * 
            np.sqrt(0.5 / (trace_fraction * hessian_privacy))
        )
        noisy_trace = trace_coefficient * max(
            np.trace(hessian) + np.random.normal(scale=trace_noise_scale),
            0
        )
        
        # Compute minimum eigenvalue based on noisy trace
        min_eigenvalue = max(
            (noisy_trace / (n_samples**2 * (1 - trace_fraction) * hessian_privacy))**(1/3),
            1.0 / n_samples
        )
        
        gradient_norm = np.linalg.norm(noisy_gradient)
        
        if self.regularization_type == "add":
            # Add regularization to diagonal
            sensitivity = (
                gradient_norm * SMOOTHNESS_PARAMETER / 
                (n_samples * min_eigenvalue**2 + SMOOTHNESS_PARAMETER * min_eigenvalue)
            )
            noise_scale = sensitivity * np.sqrt(0.5 / ((1 - trace_fraction) * hessian_privacy))
            direction_noise = np.random.normal(scale=noise_scale, size=n_features)
            
            regularized_hessian = hessian + min_eigenvalue * np.eye(n_features)
            newton_direction = np.linalg.solve(regularized_hessian, noisy_gradient)
            
            return current_weights - newton_direction + direction_noise
            
        else:  # regularization_type == "clip"
            # Clip eigenvalues
            sensitivity = (
                gradient_norm * SMOOTHNESS_PARAMETER / 
                (n_samples * min_eigenvalue**2 - SMOOTHNESS_PARAMETER * min_eigenvalue)
            )
            noise_scale = sensitivity * np.sqrt(0.5 / ((1 - trace_fraction) * hessian_privacy))
            direction_noise = np.random.normal(scale=noise_scale, size=n_features)
            
            # Eigendecomposition and clipping
            eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            large_eigenvalues = eigenvalues[eigenvalues >= min_eigenvalue]
            n_large = len(large_eigenvalues)
            
            if n_large == 0:
                # All eigenvalues are small
                hessian_inverse = (1.0 / min_eigenvalue) * np.eye(n_features)
            else:
                # Project onto large eigenvalue subspace
                large_eigenvectors = eigenvectors[:, -n_large:]
                hessian_inverse = (
                    np.dot(
                        large_eigenvectors * (1.0 / large_eigenvalues - 1.0 / min_eigenvalue),
                        large_eigenvectors.T
                    ) + 
                    (1.0 / min_eigenvalue) * np.eye(n_features)
                )
            
            newton_direction = hessian_inverse @ noisy_gradient
            return current_weights - newton_direction + direction_noise
    
    def _compute_noisy_direction_stochastic(
        self,
        current_weights: np.ndarray,
        logistic_model: MyLogisticRegression,
        hyperparameters: Dict,
        noisy_gradient: np.ndarray
    ) -> np.ndarray:
        """Compute Newton direction with noise for privacy (stochastic).
        
        Args:
            current_weights: Current parameter vector
            logistic_model: Logistic regression model
            hyperparameters: Algorithm hyperparameters
            noisy_gradient: Previously computed noisy gradient
            
        Returns:
            Next weight vector
        """
        n_samples = logistic_model.num_samples
        n_features = logistic_model.dim
        
        # Extract hyperparameters
        hessian_noise_std = hyperparameters["noise_multiplier_hess"]
        hessian_batch_prob = hyperparameters["batchsize_hess"] / n_samples
        min_eigenvalue = hyperparameters["min_eval"]
        
        # Sample mini-batch for Hessian
        sample_indicators = np.random.binomial(
            n=1, 
            p=hessian_batch_prob, 
            size=n_samples
        )
        batch_indices = np.where(sample_indicators == 1)[0]
        actual_batch_size = len(batch_indices)
        
        # Compute mini-batch Hessian
        hessian = (
            actual_batch_size / (n_samples * hessian_batch_prob) * 
            self.hessian_function(current_weights, batch_indices)
        )
        
        gradient_norm = np.linalg.norm(noisy_gradient)
        
        if self.regularization_type == "add":
            # Add regularization
            sensitivity = (
                gradient_norm * SMOOTHNESS_PARAMETER / 
                (n_samples * hessian_batch_prob * min_eigenvalue**2 + 
                 SMOOTHNESS_PARAMETER * min_eigenvalue)
            )
            direction_noise = np.random.normal(
                scale=sensitivity * hessian_noise_std, 
                size=n_features
            )
            
            regularized_hessian = hessian + min_eigenvalue * np.eye(n_features)
            newton_direction = np.linalg.solve(regularized_hessian, noisy_gradient)
            
            return current_weights - newton_direction + direction_noise
            
        else:  # regularization_type == "clip"
            # Ensure minimum eigenvalue is not too small
            effective_min = max(min_eigenvalue, 1.0 / (n_samples * hessian_batch_prob))
            
            sensitivity = (
                gradient_norm * SMOOTHNESS_PARAMETER / 
                (n_samples * hessian_batch_prob * effective_min**2 - 
                 SMOOTHNESS_PARAMETER * effective_min)
            )
            direction_noise = np.random.normal(
                scale=sensitivity * hessian_noise_std, 
                size=n_features
            )
            
            # Eigendecomposition and clipping
            eigenvalues, eigenvectors = np.linalg.eigh(hessian)
            large_eigenvalues = eigenvalues[eigenvalues >= effective_min]
            n_large = len(large_eigenvalues)
            
            if n_large == 0:
                hessian_inverse = (1.0 / effective_min) * np.eye(n_features)
            else:
                large_eigenvectors = eigenvectors[:, -n_large:]
                hessian_inverse = (
                    np.dot(
                        large_eigenvectors * (1.0 / large_eigenvalues - 1.0 / effective_min),
                        large_eigenvectors.T
                    ) + 
                    (1.0 / effective_min) * np.eye(n_features)
                )
            
            newton_direction = hessian_inverse @ noisy_gradient
            return current_weights - newton_direction + direction_noise
    

    update_rule = update_full_batch
    update_rule_stochastic = update_stochastic
    noisy_grad = _compute_noisy_gradient
    noisy_direction = _compute_noisy_direction
    noisy_direction_stochastic = _compute_noisy_direction_stochastic



CompareAlgs = AlgorithmComparison
private_newton = differentially_private_newton
eigenclip = eigenvalue_clipping
gd_priv = differentially_private_gradient_descent
sgd_priv = differentially_private_sgd
gd_priv_optls = dp_gd_with_line_search
backtracking_ls = backtracking_line_search
newton = newton_method_non_private
DoubleNoiseMech = DoubleNoiseMechanism
