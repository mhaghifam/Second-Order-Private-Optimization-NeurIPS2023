
"""Dataset loader for logistic regression experiments with various datasets."""

import os
import ssl
import tarfile
import urllib.request
from typing import Tuple, Optional

import numpy as np
import requests
import torch
from sklearn import preprocessing
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from torchvision import datasets, transforms

from my_logistic_regression import MyLogisticRegression

# Directory configuration
DATASETS_BASE_PATH = './src/datasets_directory'
DATA_DIRECTORY = f'{DATASETS_BASE_PATH}/data'
CACHE_DIRECTORY = f'{DATASETS_BASE_PATH}/cache_datasets'

# Optimization parameters
DEFAULT_REGULARIZATION = 1e-9
NEWTON_MAX_ITERATIONS = 8
LINE_SEARCH_MIN_STEP_SIZE = 1e-6

# Line search parameters (Armijo-Goldstein conditions)
ARMIJO_ALPHA = 0.4  # Sufficient decrease parameter
BACKTRACK_BETA = 0.95  # Step size reduction factor

# Dataset specific constants
FMNIST_LABEL_TSHIRT = 0
FMNIST_LABEL_DRESS = 3
PROTEIN_SAMPLE_SIZE = 50000
PROTEIN_RANDOM_SEED = 3000

# SSL configuration for dataset downloads
ssl._create_default_https_context = ssl._create_unverified_context


def normalize_feature_vectors(feature_matrix: np.ndarray) -> np.ndarray:
    """Normalize feature vectors to zero mean and unit variance.
    
    Args:
        feature_matrix: Input feature matrix of shape (n_samples, n_features)
        
    Returns:
        Normalized feature matrix with same shape as input
    """
    feature_mean = np.mean(feature_matrix, axis=0)
    feature_std = np.std(feature_matrix, axis=0)
    normalized_features = (feature_matrix - feature_mean) / feature_std
    return normalized_features


def backtracking_line_search(
    logistic_model: MyLogisticRegression,
    search_direction: np.ndarray,
    current_weights: np.ndarray,
    alpha: float = ARMIJO_ALPHA,
    beta: float = BACKTRACK_BETA
) -> float:
    """Find optimal step size using backtracking line search with Armijo condition.
    
    Implements backtracking to find a step size that satisfies the Armijo
    sufficient decrease condition for the objective function.
    
    Args:
        logistic_model: Logistic regression model with loss and gradient methods
        search_direction: Direction vector for optimization step (e.g., gradient)
        current_weights: Current parameter vector
        alpha: Sufficient decrease parameter (default 0.4)
        beta: Step size reduction factor (default 0.95)
        
    Returns:
        Optimal step size satisfying Armijo condition
    """
    step_size = 100.0
    initial_loss = logistic_model.loss(current_weights)
    gradient_dot_direction = np.dot(search_direction, logistic_model.grad(current_weights))
    
    # Backtrack until Armijo condition is satisfied
    while (logistic_model.loss(current_weights - step_size * search_direction) 
           >= initial_loss - step_size * alpha * gradient_dot_direction):
        step_size *= beta
        
        # Prevent infinite loop with minimum step size
        if step_size < LINE_SEARCH_MIN_STEP_SIZE:
            break
            
    return step_size


def newton_method_optimizer(
    dataset: Tuple[np.ndarray, np.ndarray],
    initial_weights: np.ndarray,
    add_bias: bool = True
) -> np.ndarray:
    """Optimize logistic regression using Newton's method with line search.
    
    Implements Newton's method with backtracking line search for finding
    optimal parameters of a logistic regression model.
    
    Args:
        dataset: Tuple of (feature_matrix, labels)
        initial_weights: Initial parameter vector
        add_bias: Whether to add bias term to features (default True)
        
    Returns:
        Optimized weight vector
    """
    feature_matrix, labels = dataset
    
    # Add bias column if requested
    if add_bias:
        n_samples = feature_matrix.shape[0]
        bias_column = np.ones((n_samples, 1))
        feature_matrix = np.hstack((bias_column, feature_matrix))
    
    # Initialize logistic regression model
    logistic_model = MyLogisticRegression(
        feature_matrix, 
        labels, 
        reg=DEFAULT_REGULARIZATION
    )
    
    current_weights = initial_weights
    
    # Newton iterations with line search
    for iteration in range(NEWTON_MAX_ITERATIONS):
        # Compute Newton direction: H^(-1) * gradient
        hessian_matrix = logistic_model.hess(current_weights)
        gradient_vector = logistic_model.grad_wor(current_weights)
        newton_direction = np.linalg.solve(hessian_matrix, gradient_vector)
        
        # Find optimal step size
        step_size = backtracking_line_search(
            logistic_model, 
            newton_direction, 
            current_weights
        )
        
        # Update weights
        current_weights = current_weights - step_size * newton_direction
    
    # Return best weights (current or initial based on loss)
    if logistic_model.loss_wor(current_weights) < logistic_model.loss_wor(initial_weights):
        return current_weights
    else:
        return initial_weights


class DatasetManager:
    """Manages loading and preprocessing of various datasets for experiments.
    
    This class provides methods to load different datasets (synthetic and real)
    and find optimal classifiers for logistic regression experiments.
    """
    
    def __init__(self):
        """Initialize dataset manager and create necessary directories."""
        self._create_directories()
    
    def _create_directories(self):
        """Create data and cache directories if they don't exist."""
        os.makedirs(DATA_DIRECTORY, exist_ok=True)
        os.makedirs(CACHE_DIRECTORY, exist_ok=True)
    
    def find_optimal_classifier(
        self, 
        dataset: Tuple[np.ndarray, np.ndarray], 
        use_bias: bool = True
    ) -> np.ndarray:
        """Find optimal weight vector for logistic regression.
        
        Uses sklearn's LogisticRegression for initialization, then refines
        with Newton's method for better convergence.
        
        Args:
            dataset: Tuple of (feature_matrix, labels)
            use_bias: Whether to include bias term (default True)
            
        Returns:
            Optimal weight vector
        """
        feature_matrix, labels = dataset
        
        # Initialize with sklearn's logistic regression
        sklearn_model = LogisticRegression(
            max_iter=200,
            fit_intercept=use_bias,
            C=1 / DEFAULT_REGULARIZATION
        )
        sklearn_model.fit(feature_matrix, labels)
        
        # Extract initial weights from sklearn model
        if use_bias:
            initial_weights = np.concatenate([
                sklearn_model.intercept_, 
                np.squeeze(sklearn_model.coef_)
            ])
        else:
            initial_weights = np.squeeze(sklearn_model.coef_)
        
        # Refine with Newton's method
        optimal_weights = newton_method_optimizer(
            dataset, 
            initial_weights, 
            add_bias=use_bias
        )
        
        return optimal_weights
    
    def load_fashion_mnist(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess Fashion-MNIST dataset for binary classification.
        
        Extracts T-shirt/top (label 0) and Dress (label 3) classes for
        binary classification task.
        
        Returns:
            Tuple of (features, labels, optimal_weights)
        """
        # Define data transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Load Fashion-MNIST training data
        train_dataset = datasets.FashionMNIST(
            root=DATA_DIRECTORY,
            download=True,
            train=True,
            transform=transform
        )
        
        # Load entire dataset at once
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=len(train_dataset)
        )
        
        # Extract features and labels
        data_batch = next(iter(train_loader))
        features = data_batch[0].numpy()
        features = features.reshape(len(features), -1)  # Flatten images
        labels = data_batch[1].numpy()
        
        # Select two classes for binary classification
        class_0_indices = np.nonzero(labels == FMNIST_LABEL_TSHIRT)[0]
        class_1_indices = np.nonzero(labels == FMNIST_LABEL_DRESS)[0]
        
        # Convert to binary labels (-1, 1)
        binary_labels = labels.copy()
        binary_labels[class_0_indices] = -1
        binary_labels[class_1_indices] = 1
        
        # Select only samples from chosen classes
        selected_indices = np.concatenate((class_0_indices, class_1_indices))
        features = features[selected_indices]
        binary_labels = binary_labels[selected_indices]
        
        # Find optimal classifier
        dataset = (features, binary_labels)
        optimal_weights = self.find_optimal_classifier(dataset, use_bias=False)
        
        return features, binary_labels, optimal_weights
    
    def load_a1a_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess a1a dataset from LIBSVM repository.
        
        Returns:
            Tuple of (features_with_bias, labels, optimal_weights)
        """
        a1a_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t'
        data_path = f'{DATA_DIRECTORY}/a1a'
        
        # Download dataset if not exists
        if not os.path.exists(data_path):
            urllib.request.urlretrieve(a1a_url, data_path)
        
        # Load dataset in LIBSVM format
        features_sparse, labels = sklearn.datasets.load_svmlight_file(data_path)
        features = features_sparse.toarray()
        
        # Standardize features
        scaler = preprocessing.StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Convert labels to float
        labels = labels.astype(float)
        
        # Find optimal classifier
        dataset = (features_normalized, labels)
        optimal_weights = self.find_optimal_classifier(dataset, use_bias=True)
        
        # Add bias column to features
        n_samples = features_normalized.shape[0]
        features_with_bias = np.hstack((
            np.ones((n_samples, 1)), 
            features_normalized
        ))
        
        return features_with_bias, labels, optimal_weights
    
    def load_protein_dataset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and preprocess protein homology dataset from KDD Cup 2004.
        
        Returns:
            Tuple of (features_with_bias, labels, optimal_weights)
        """
        protein_dir = f'{DATA_DIRECTORY}/protein'
        
        # Download and extract if not exists
        if not os.path.exists(protein_dir):
            os.makedirs(protein_dir)
            self._download_protein_dataset(protein_dir)
        
        # Load data
        data_file = f'{protein_dir}/bio_train.dat'
        full_data = np.loadtxt(data_file)
        features = full_data[:, 3:]  # Features start from column 3
        labels_raw = full_data[:, 2]  # Labels in column 2
        
        # Convert to binary labels
        labels = labels_raw.copy()
        labels[labels_raw == 0] = -1
        labels[labels_raw == 1] = 1
        
        # Random sampling for computational efficiency
        np.random.seed(PROTEIN_RANDOM_SEED)
        n_total = len(features)
        sample_indices = np.random.choice(n_total, PROTEIN_SAMPLE_SIZE, replace=False)
        np.random.seed(None)  # Reset random seed
        
        features_sampled = features[sample_indices]
        labels_sampled = labels[sample_indices]
        
        # Normalize features
        features_normalized = normalize_feature_vectors(features_sampled)
        
        # Find optimal classifier
        dataset = (features_normalized, labels_sampled)
        optimal_weights = self.find_optimal_classifier(dataset, use_bias=True)
        
        # Add bias column
        n_samples = features_normalized.shape[0]
        features_with_bias = np.hstack((
            np.ones((n_samples, 1)), 
            features_normalized
        ))
        
        return features_with_bias, labels_sampled, optimal_weights
    
    def _download_protein_dataset(self, target_dir: str):
        """Download and extract protein dataset.
        
        Args:
            target_dir: Directory to save the dataset
        """
        protein_url = 'https://kdd.org/cupfiles/KDDCupData/2004/data_kddcup04.tar.gz'
        archive_path = f'{target_dir}/data_kddcup04.tar.gz'
        
        # Download dataset
        response = requests.get(protein_url, stream=True, timeout=100)
        if response.status_code == 200:
            with open(archive_path, 'wb') as file:
                file.write(response.raw.read())
        
        # Extract archive
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(target_dir)
    
    def generate_synthetic_dataset(
        self, 
        n_samples: int = 10000, 
        n_features: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic dataset for logistic regression experiments.
        
        Creates a synthetic dataset with unit-norm features and labels
        sampled from a logistic distribution.
        
        Args:
            n_samples: Number of samples to generate (default 10000)
            n_features: Number of features per sample (default 100)
            
        Returns:
            Tuple of (features, labels, optimal_weights)
        """
        # Generate features from multivariate normal
        mean = np.zeros(n_features)
        covariance = np.eye(n_features)
        features_raw = np.random.multivariate_normal(mean, covariance, n_samples)
        
        # Normalize to unit vectors
        norms = np.linalg.norm(features_raw, axis=1)
        features_normalized = features_raw / norms[:, np.newaxis]
        
        # Generate true weight vector
        true_weights = np.ones(n_features)
        true_weights[0] = 1  # Set first weight differently
        
        # Generate labels from logistic distribution
        logits = np.dot(features_normalized, true_weights)
        probabilities = np.exp(logits) / (1 + np.exp(logits))
        binary_outcomes = np.random.binomial(1, probabilities)
        labels = 2 * binary_outcomes - 1  # Convert to {-1, 1}
        
        # Find optimal classifier
        dataset = (features_normalized, labels)
        optimal_weights = self.find_optimal_classifier(dataset, use_bias=False)
        
        return features_normalized, labels, optimal_weights
