# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Run and compare differentially private optimization algorithms.

This module provides functionality to:
1. Run various DP optimization algorithms on different datasets
2. Compare their performance across multiple metrics
3. Save results for analysis

Usage:
    python run.py --datasetname a1a --alg_type double_noise --total 1.0 \
                  --numiter 100 --grad_frac 0.5 --trace_frac 0.5 --trace_coeff 1.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Callable, Optional, Any
import numpy as np
from scipy.optimize import fsolve

from dataset_loader import Mydatasets
from my_logistic_regression import MyLogisticRegression
from opt_algs import (
    CompareAlgs, 
    DoubleNoiseMech, 
    gd_priv, 
    private_newton
)

# Configuration constants
RESULTS_DIRECTORY = Path("src/results")
DEFAULT_NUM_REPETITIONS = 10
DELTA_FACTOR_EXPONENT = 2  # Delta = (1/n)^DELTA_FACTOR_EXPONENT

# Algorithm type identifiers
ALG_TYPE_DOUBLE_NOISE = "double_noise"
ALG_TYPE_DP_GD = "dp_gd"
ALG_TYPE_DAMPED_NEWTON = "damped_newton"

# Initial value for root finding in privacy conversion
ZCDP_ROOT_INITIAL_GUESS = 0.001


class PrivacyConverter:
    """Utility class for converting between different privacy definitions.
    
    Handles conversions between:
    - (ε, δ)-differential privacy
    - ρ-zero concentrated differential privacy (zCDP)
    
    Based on Lemma 3.6 from [BS16]: "Concentrated Differential Privacy:
    Simplifications, Extensions, and Lower Bounds" by Bun & Steinke.
    """
    
    @staticmethod
    def zcdp_to_epsilon_delta(rho: float, delta: float) -> float:
        """Convert zCDP parameter to (ε,δ)-DP.
        
        Uses the formula: ε = ρ + √(4ρ * log(√(πρ) / δ))
        
        Args:
            rho: zCDP privacy parameter
            delta: Target δ value for (ε,δ)-DP
            
        Returns:
            Corresponding ε value for (ε,δ)-DP
        """
        epsilon = rho + np.sqrt(4 * rho * np.log(np.sqrt(np.pi * rho) / delta))
        return epsilon
    
    @staticmethod
    def epsilon_delta_to_zcdp(epsilon: float, delta: float) -> float:
        """Convert (ε,δ)-DP to zCDP parameter.
        
        Finds ρ such that the zCDP guarantee corresponds to (ε,δ)-DP.
        Uses numerical root finding since the inverse is not closed-form.
        
        Args:
            epsilon: ε value from (ε,δ)-DP
            delta: δ value from (ε,δ)-DP
            
        Returns:
            Corresponding ρ value for ρ-zCDP
        """
        def objective_function(rho_value):
            return PrivacyConverter.zcdp_to_epsilon_delta(rho_value, delta) - epsilon
        
        # Use scipy's fsolve to find the root
        solution = fsolve(objective_function, x0=ZCDP_ROOT_INITIAL_GUESS)
        rho = solution[-1]
        
        # Validate the solution
        if rho <= 0:
            raise ValueError(f"Invalid zCDP parameter computed: {rho}")
        
        return rho


class AlgorithmFactory:
    """Factory class for creating optimization algorithm instances.
    
    This class handles the creation and configuration of different
    optimization algorithms based on the specified type and parameters.
    """
    
    @staticmethod
    def create_algorithms(
        algorithm_type: str,
        logistic_model: MyLogisticRegression,
        experiment_params: Dict[str, Any]
    ) -> Dict[str, Callable]:
        """Create algorithm update functions based on type.
        
        Args:
            algorithm_type: Type of algorithms to create
            logistic_model: Logistic regression model instance
            experiment_params: Hyperparameters for the algorithms
            
        Returns:
            Dictionary mapping algorithm names to update functions
            
        Raises:
            ValueError: If algorithm type is not recognized
        """
        if algorithm_type == ALG_TYPE_DOUBLE_NOISE:
            return AlgorithmFactory._create_double_noise_algorithms(logistic_model)
        elif algorithm_type == ALG_TYPE_DP_GD:
            return {"DPGD": gd_priv}
        elif algorithm_type == ALG_TYPE_DAMPED_NEWTON:
            return {"private-newton": private_newton}
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    @staticmethod
    def _create_double_noise_algorithms(
        logistic_model: MyLogisticRegression
    ) -> Dict[str, Callable]:
        """Create all variants of the double noise mechanism.
        
        Args:
            logistic_model: Logistic regression model instance
            
        Returns:
            Dictionary with four DN variants
        """
        # Create four variants: {Hessian, UpperBound} x {Add, Clip}
        algorithms = {
            "DN-Hess-add": DoubleNoiseMech(
                logistic_model, 
                type_reg="add", 
                curvature_info="hessian"
            ).update_rule,
            "DN-Hess-clip": DoubleNoiseMech(
                logistic_model, 
                type_reg="clip", 
                curvature_info="hessian"
            ).update_rule,
            "DN-UB-add": DoubleNoiseMech(
                logistic_model, 
                type_reg="add", 
                curvature_info="ub"
            ).update_rule,
            "DN-UB-clip": DoubleNoiseMech(
                logistic_model, 
                type_reg="clip", 
                curvature_info="ub"
            ).update_rule,
        }
        return algorithms
    
    @staticmethod
    def generate_filename(
        algorithm_type: str,
        dataset_name: str,
        privacy_epsilon: float,
        experiment_params: Dict[str, Any]
    ) -> str:
        """Generate descriptive filename for results.
        
        Args:
            algorithm_type: Type of algorithm
            dataset_name: Name of dataset
            privacy_epsilon: Privacy budget (epsilon)
            experiment_params: Experiment hyperparameters
            
        Returns:
            Formatted filename string
        """
        base_parts = [
            algorithm_type.replace("_", ""),
            dataset_name,
            f"{privacy_epsilon}DP",
            f"iter{experiment_params['num_iteration']}"
        ]
        
        # Add algorithm-specific parameters
        if algorithm_type == ALG_TYPE_DOUBLE_NOISE:
            base_parts.extend([
                f"gf{experiment_params['grad_frac']}",
                f"tf{experiment_params['trace_frac']}",
                f"tc{experiment_params['trace_coeff']}"
            ])
        elif algorithm_type == ALG_TYPE_DAMPED_NEWTON:
            base_parts.append(f"gf{experiment_params['grad_frac']}")
        
        filename = "_".join(base_parts) + ".json"
        return filename


class ExperimentRunner:
    """Manages execution and statistics collection for optimization experiments.
    
    This class handles:
    - Running multiple repetitions of algorithms
    - Collecting performance metrics
    - Computing statistics across repetitions
    
    Attributes:
        comparison_framework: CompareAlgs instance for running algorithms
        algorithms: Dictionary of algorithm update functions
        num_repetitions: Number of times to repeat each experiment
        collected_metrics: Dictionary storing all collected metrics
    """
    
    def __init__(
        self,
        comparison_framework: CompareAlgs,
        algorithms: Dict[str, Callable],
        num_repetitions: int = DEFAULT_NUM_REPETITIONS
    ):
        """Initialize experiment runner.
        
        Args:
            comparison_framework: Framework for comparing algorithms
            algorithms: Dictionary mapping names to update functions
            num_repetitions: Number of experimental repetitions
        """
        self.comparison_framework = comparison_framework
        self.algorithms = algorithms
        self.num_repetitions = num_repetitions
        
        # Initialize metric storage
        self.collected_metrics = {
            'losses': {},
            'gradient_norms': {},
            'accuracies': {},
            'wall_times': {}
        }
    
    def run_experiments(self) -> None:
        """Execute all algorithms for the specified number of repetitions."""
        print(f"Running {self.num_repetitions} repetitions...")
        
        for repetition in range(self.num_repetitions):
            if repetition % max(1, self.num_repetitions // 10) == 0:
                print(f"  Repetition {repetition + 1}/{self.num_repetitions}")
            
            # Run each algorithm
            for algorithm_name, update_function in self.algorithms.items():
                self.comparison_framework.add_algo(update_function, algorithm_name)
            
            # Collect metrics for this repetition
            self._collect_repetition_metrics(repetition)
    
    def _collect_repetition_metrics(self, repetition: int) -> None:
        """Collect metrics from one repetition.
        
        Args:
            repetition: Current repetition number
        """
        # Get metrics from comparison framework
        losses = self.comparison_framework.loss_vals()
        gradient_norms = self.comparison_framework.gradnorm_vals()
        accuracies = self.comparison_framework.accuracy_vals()
        wall_times = self.comparison_framework.wall_clock_alg()
        
        # Store or append metrics
        if repetition == 0:
            self.collected_metrics['losses'] = losses
            self.collected_metrics['gradient_norms'] = gradient_norms
            self.collected_metrics['accuracies'] = accuracies
            self.collected_metrics['wall_times'] = wall_times
        else:
            # Append to existing metrics
            for algorithm_name in self.collected_metrics['losses']:
                self.collected_metrics['losses'][algorithm_name].extend(
                    losses[algorithm_name]
                )
                self.collected_metrics['gradient_norms'][algorithm_name].extend(
                    gradient_norms[algorithm_name]
                )
                self.collected_metrics['accuracies'][algorithm_name].extend(
                    accuracies[algorithm_name]
                )
                self.collected_metrics['wall_times'][algorithm_name].extend(
                    wall_times[algorithm_name]
                )
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics across all repetitions.
        
        Returns:
            Dictionary containing mean and standard error for all metrics
        """
        self.run_experiments()
        
        results = {
            "optimal_accuracy": self.comparison_framework.accuracy_np().tolist()
        }
        
        # Compute statistics for each algorithm
        for algorithm_name in self.collected_metrics['losses']:
            results[algorithm_name] = self._compute_algorithm_statistics(algorithm_name)
        
        return results
    
    def _compute_algorithm_statistics(self, algorithm_name: str) -> Dict[str, Any]:
        """Compute statistics for a single algorithm.
        
        Args:
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary with mean and standard error for each metric
        """
        sqrt_n = np.sqrt(self.num_repetitions)
        
        # Convert to numpy arrays for easier computation
        losses = np.array(self.collected_metrics['losses'][algorithm_name])
        gradient_norms = np.array(self.collected_metrics['gradient_norms'][algorithm_name])
        accuracies = np.array(self.collected_metrics['accuracies'][algorithm_name])
        wall_times = np.array(self.collected_metrics['wall_times'][algorithm_name])
        
        # Compute mean and standard error
        statistics = {
            "loss_avg": np.mean(losses, axis=0).tolist(),
            "loss_std": (np.std(losses, axis=0) / sqrt_n).tolist(),
            "gradnorm_avg": np.mean(gradient_norms, axis=0).tolist(),
            "gradnorm_std": np.std(gradient_norms, axis=0).tolist(),
            "acc_avg": np.mean(accuracies, axis=0).tolist(),
            "acc_std": (np.std(accuracies, axis=0) / sqrt_n).tolist(),
            "clock_time_avg": np.mean(wall_times, axis=0).tolist(),
            "clock_time_std": (np.std(wall_times, axis=0) / sqrt_n).tolist()
        }
        
        return statistics


def run_experiment(
    dataset_name: str,
    algorithm_type: str,
    experiment_parameters: Dict[str, Any]
) -> None:
    """Main function to run a single experiment configuration.
    
    Args:
        dataset_name: Name of the dataset to use
        algorithm_type: Type of optimization algorithm
        experiment_parameters: Dictionary of hyperparameters
    """
    print(f"\n{'='*60}")
    print(f"Running Experiment")
    print(f"  Dataset: {dataset_name}")
    print(f"  Algorithm Type: {algorithm_type}")
    print(f"  Privacy Budget (ε): {experiment_parameters['total']}")
    print(f"  Iterations: {experiment_parameters['num_iteration']}")
    print(f"{'='*60}\n")
    
    # Load dataset
    print("Loading dataset...")
    dataset_loader = Mydatasets()
    features, labels, optimal_weights = getattr(dataset_loader, dataset_name)()
    n_samples = len(labels)
    print(f"  Dataset size: {n_samples} samples")
    
    # Convert privacy parameters
    epsilon_dp = experiment_parameters["total"]
    delta_dp = (1.0 / n_samples) ** DELTA_FACTOR_EXPONENT
    rho_zcdp = PrivacyConverter.epsilon_delta_to_zcdp(epsilon_dp, delta_dp)
    experiment_parameters["total"] = rho_zcdp
    print(f"  Privacy: ε={epsilon_dp}, δ={delta_dp:.2e}, ρ={rho_zcdp:.4f}")
    
    # Create logistic regression model
    logistic_model = MyLogisticRegression(features, labels)
    
    # Create algorithms
    print("\nInitializing algorithms...")
    algorithms = AlgorithmFactory.create_algorithms(
        algorithm_type,
        logistic_model,
        experiment_parameters
    )
    print(f"  Algorithms: {', '.join(algorithms.keys())}")
    
    # Generate output filename
    output_filename = AlgorithmFactory.generate_filename(
        algorithm_type,
        dataset_name,
        epsilon_dp,
        experiment_parameters
    )
    output_path = RESULTS_DIRECTORY / output_filename
    
    # Create comparison framework
    comparison_framework = CompareAlgs(
        logistic_model,
        optimal_weights,
        experiment_parameters
    )
    
    # Run experiments and collect statistics
    print("\nRunning experiments...")
    experiment_runner = ExperimentRunner(
        comparison_framework,
        algorithms,
        num_repetitions=DEFAULT_NUM_REPETITIONS
    )
    results = experiment_runner.compute_statistics()
    results["num_samples"] = n_samples
    
    # Save results
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    print(f"  Final losses: {[f'{alg}: {results[alg]['loss_avg'][-1]:.6f}' for alg in algorithms]}")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Run differentially private optimization experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--datasetname",
        required=True,
        choices=["a1a_dataset", "protein_dataset", "fmnist_dataset", "synthetic_dataset"],
        help="Dataset to use for experiments"
    )
    parser.add_argument(
        "--alg_type",
        required=True,
        choices=[ALG_TYPE_DOUBLE_NOISE, ALG_TYPE_DP_GD, ALG_TYPE_DAMPED_NEWTON],
        help="Type of optimization algorithm"
    )
    parser.add_argument(
        "--total",
        type=float,
        required=True,
        help="Total privacy budget (epsilon)"
    )
    parser.add_argument(
        "--numiter",
        type=int,
        required=True,
        help="Number of optimization iterations"
    )
    
    # Algorithm-specific optional arguments
    parser.add_argument(
        "--grad_frac",
        type=float,
        help="Fraction of privacy budget for gradient (DN and Newton)"
    )
    parser.add_argument(
        "--trace_frac",
        type=float,
        help="Fraction of Hessian budget for trace (DN only)"
    )
    parser.add_argument(
        "--trace_coeff",
        type=float,
        help="Coefficient for trace regularization (DN only)"
    )
    
    return parser.parse_args()


def validate_and_prepare_parameters(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate arguments and prepare hyperparameters dictionary.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Dictionary of hyperparameters
        
    Raises:
        ValueError: If required parameters are missing
    """
    hyperparameters = {
        "total": args.total,
        "num_iteration": args.numiter
    }
    
    if args.alg_type == ALG_TYPE_DOUBLE_NOISE:
        # Double noise requires all parameters
        if not all([args.grad_frac, args.trace_frac, args.trace_coeff]):
            raise ValueError(
                "Double noise algorithm requires --grad_frac, --trace_frac, and --trace_coeff"
            )
        hyperparameters.update({
            "grad_frac": args.grad_frac,
            "trace_frac": args.trace_frac,
            "trace_coeff": args.trace_coeff
        })
        
    elif args.alg_type == ALG_TYPE_DAMPED_NEWTON:
        # Damped Newton requires grad_frac
        if args.grad_frac is None:
            raise ValueError("Damped Newton algorithm requires --grad_frac")
        hyperparameters["grad_frac"] = args.grad_frac
    
    # DP-GD doesn't need additional parameters
    
    return hyperparameters


def main():
    """Main entry point for the experiment runner."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Validate and prepare parameters
        hyperparameters = validate_and_prepare_parameters(args)
        
        # Run experiment
        run_experiment(
            dataset_name=args.datasetname,
            algorithm_type=args.alg_type,
            experiment_parameters=hyperparameters
        )
        
        print("\n✓ Experiment completed successfully!")
        
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()