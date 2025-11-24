
"""Logistic regression implementation with L2 regularization and optimization utilities."""

from typing import Optional, Union
import numpy as np

# Numerical stability constants
EPSILON = 1e-9
DEFAULT_REGULARIZATION = 1e-8
HALF = 0.5
QUARTER = 0.25
CLASSIFICATION_THRESHOLD = 0.5


class MyLogisticRegression:
    """Logistic regression model with L2 regularization.
    
    This class implements logistic regression for binary classification with:
    - Features normalized to have norm <= 1
    - Binary labels in {-1, +1}
    - L2 regularization for numerical stability
    - Methods for computing loss, gradients, and Hessian matrices
    
    Attributes:
        regularization_strength: L2 regularization coefficient
        feature_matrix: Original feature matrix (n_samples x n_features)
        labels: Binary labels array (n_samples,)
        n_samples: Number of training samples
        n_features: Number of features
        normalized_data: Feature matrix scaled by labels and normalized
    """
    
    def __init__(
        self, 
        feature_vectors: np.ndarray, 
        labels: np.ndarray, 
        reg: float = DEFAULT_REGULARIZATION
    ):
        """Initialize logistic regression model with data and regularization.
        
        The data is automatically rescaled so that ||X[i,:] * y[i]|| <= 1 for all i,
        ensuring numerical stability during optimization.
        
        Args:
            feature_vectors: Feature matrix of shape (n_samples, n_features)
            labels: Binary labels of shape (n_samples,) with values in {-1, +1}
            reg: L2 regularization coefficient (default 1e-8)
        
        Raises:
            AssertionError: If input shapes are incorrect
        """
        self.regularization_strength = float(reg)
        
        # Convert inputs to numpy arrays
        self.feature_matrix = np.array(feature_vectors)
        self.labels = np.array(labels)
        
        # Validate input shapes
        assert len(self.feature_matrix.shape) == 2, "Features must be 2D array"
        assert len(self.labels.shape) == 1, "Labels must be 1D array"
        
        # Extract dimensions
        self.n_samples, self.n_features = self.feature_matrix.shape
        assert self.labels.shape[0] == self.n_samples, "Number of labels must match samples"
        
        # Normalize data: multiply features by labels and scale to unit norm
        label_signed_features = self.feature_matrix * self.labels[:, np.newaxis]
        feature_norms = np.linalg.norm(label_signed_features, axis=1)
        scaling_factors = np.maximum(feature_norms, np.ones_like(feature_norms))
        self.normalized_data = label_signed_features / scaling_factors[:, np.newaxis]
    
    def loss(self, weights: np.ndarray) -> float:
        """Compute regularized logistic loss at given weights.
        
        The loss function is:
        L(w) = (1/n) * sum_i log(1 + exp(-y_i * <x_i, w>)) + (λ/2) * ||w||²
        
        where:
        - n is the number of samples
        - y_i is the label for sample i
        - x_i is the feature vector for sample i
        - λ is the regularization strength
        
        Args:
            weights: Parameter vector of shape (n_features,)
            
        Returns:
            Total loss (data loss + regularization loss)
        """
        # Compute logistic loss: mean of log(1 + exp(-y * x^T * w))
        linear_predictions = np.dot(self.normalized_data, weights)
        data_loss = np.mean(np.log1p(np.exp(-linear_predictions)))
        
        # Add L2 regularization term
        regularization_loss = HALF * self.regularization_strength * np.linalg.norm(weights) ** 2
        
        return data_loss + regularization_loss
    
    def loss_without_regularization(self, weights: np.ndarray) -> float:
        """Compute logistic loss without regularization term.
        
        The loss function is:
        L(w) = (1/n) * sum_i log(1 + exp(-y_i * <x_i, w>))
        
        This is useful for evaluating the pure data fit without regularization.
        
        Args:
            weights: Parameter vector of shape (n_features,)
            
        Returns:
            Data loss only (no regularization)
        """
        linear_predictions = np.dot(self.normalized_data, weights)
        data_loss = np.mean(np.log1p(np.exp(-linear_predictions)))
        return data_loss
    
    def accuracy(self, weights: np.ndarray) -> float:
        """Compute classification accuracy of the model.
        
        Uses the standard logistic regression decision rule:
        - Predict +1 if P(y=1|x) >= 0.5
        - Predict -1 otherwise
        
        Args:
            weights: Parameter vector of shape (n_features,)
            
        Returns:
            Fraction of correctly classified samples
        """
        # Compute scores for original (unnormalized) features
        scores = np.dot(self.feature_matrix, weights)
        
        # Compute probabilities using numerically stable sigmoid
        probabilities = np.where(
            scores >= 0,
            1 / (1 + np.exp(-scores)),  # For positive scores
            np.exp(scores) / (1 + np.exp(scores))  # For negative scores
        )
        
        # Make predictions based on probability threshold
        predictions = np.where(probabilities >= CLASSIFICATION_THRESHOLD, 1, -1)
        
        # Calculate accuracy
        correct_predictions = (predictions == self.labels)
        return np.mean(correct_predictions)
    
    def gradient(
        self, 
        weights: np.ndarray, 
        batch_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute gradient of regularized loss with respect to weights.
        
        The gradient is:
        ∇L(w) = -(1/n) * sum_i (y_i * x_i) / (1 + exp(y_i * <x_i, w>)) + λ * w
        
        Args:
            weights: Parameter vector of shape (n_features,)
            batch_indices: Optional indices for mini-batch gradient computation
            
        Returns:
            Gradient vector of shape (n_features,)
        """
        # Select batch or use full data
        if batch_indices is not None:
            data_batch = self.normalized_data[batch_indices]
        else:
            data_batch = self.normalized_data
        
        # Compute gradient coefficients: -1 / (1 + exp(y * x^T * w))
        linear_predictions = np.dot(data_batch, weights)
        gradient_coefficients = -1 / (1 + np.exp(linear_predictions))
        
        # Compute data gradient: mean of coefficient * data
        data_gradient = np.mean(
            data_batch * gradient_coefficients[:, np.newaxis], 
            axis=0
        )
        
        # Add regularization gradient
        total_gradient = data_gradient + self.regularization_strength * weights
        
        return total_gradient
    
    def gradient_without_regularization(
        self, 
        weights: np.ndarray, 
        batch_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute gradient of loss without regularization term.
        
        The gradient is:
        ∇L(w) = -(1/n) * sum_i (y_i * x_i) / (1 + exp(y_i * <x_i, w>))
        
        Args:
            weights: Parameter vector of shape (n_features,)
            batch_indices: Optional indices for mini-batch gradient computation
            
        Returns:
            Gradient vector of shape (n_features,) without regularization
        """
        # Select batch or use full data
        if batch_indices is not None:
            data_batch = self.normalized_data[batch_indices]
        else:
            data_batch = self.normalized_data
        
        # Compute gradient coefficients
        linear_predictions = np.dot(data_batch, weights)
        gradient_coefficients = -1 / (1 + np.exp(linear_predictions))
        
        # Compute and return data gradient only
        data_gradient = np.mean(
            data_batch * gradient_coefficients[:, np.newaxis], 
            axis=0
        )
        
        return data_gradient
    
    def hessian(
        self, 
        weights: np.ndarray, 
        batch_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute Hessian matrix (second derivatives) of regularized loss.
        
        The Hessian is:
        H(w) = (1/n) * sum_i (x_i * x_i^T) / (4 * cosh²(y_i * <x_i, w> / 2)) + λ * I
        
        This is used in second-order optimization methods like Newton's method.
        
        Args:
            weights: Parameter vector of shape (n_features,)
            batch_indices: Optional indices for mini-batch Hessian computation
            
        Returns:
            Hessian matrix of shape (n_features, n_features)
        """
        # Select batch or use full data
        if batch_indices is not None:
            data_batch = self.normalized_data[batch_indices]
            batch_size = len(batch_indices)
        else:
            data_batch = self.normalized_data
            batch_size = self.n_samples
        
        # Compute Hessian coefficients: 1 / (4 * cosh²(z/2))
        # Note: 1/cosh²(x) = 1/(exp(x) + exp(-x))²
        half_predictions = np.dot(data_batch, weights) / 2
        cosh_squared = (np.exp(half_predictions) + np.exp(-half_predictions)) ** 2
        hessian_coefficients = 1 / cosh_squared
        
        # Compute data Hessian: X^T * diag(coefficients) * X
        weighted_data = data_batch.T * hessian_coefficients
        data_hessian = np.dot(weighted_data, data_batch)
        
        # Add regularization term to diagonal
        regularization_term = self.regularization_strength * np.eye(self.n_features)
        
        return data_hessian / batch_size + regularization_term
    
    def hessian_without_regularization(
        self, 
        weights: np.ndarray, 
        batch_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute Hessian matrix without regularization term.
        
        The Hessian is:
        H(w) = (1/n) * sum_i (x_i * x_i^T) / (4 * cosh²(y_i * <x_i, w> / 2))
        
        Args:
            weights: Parameter vector of shape (n_features,)
            batch_indices: Optional indices for mini-batch Hessian computation
            
        Returns:
            Hessian matrix of shape (n_features, n_features) without regularization
        """
        # Select batch or use full data
        if batch_indices is not None:
            data_batch = self.normalized_data[batch_indices]
            batch_size = len(batch_indices)
        else:
            data_batch = self.normalized_data
            batch_size = self.n_samples
        
        # Compute Hessian coefficients
        half_predictions = np.dot(data_batch, weights) / 2
        cosh_squared = (np.exp(half_predictions) + np.exp(-half_predictions)) ** 2
        hessian_coefficients = 1 / cosh_squared
        
        # Compute and return data Hessian only
        weighted_data = data_batch.T * hessian_coefficients
        data_hessian = np.dot(weighted_data, data_batch)
        
        return data_hessian / batch_size
    
    def quadratic_upper_bound(
        self, 
        weights: np.ndarray, 
        batch_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute tightest universal quadratic upper bound on loss function.
        
        Uses the inequality:
        log(1 + exp(x)) <= log(1 + exp(a)) + (x-a)/(1 + exp(-a)) + (x-a)² * tanh(a/2)/(4*a)
        
        This provides a quadratic approximation that upper bounds the loss everywhere,
        useful for majorization-minimization algorithms.
        
        Reference: https://twitter.com/shortstein/status/1557961202256318464
        
        Args:
            weights: Parameter vector of shape (n_features,)
            batch_indices: Optional indices for mini-batch computation
            
        Returns:
            Quadratic term matrix of shape (n_features, n_features)
        """
        # Select batch or use full data
        if batch_indices is not None:
            data_batch = self.normalized_data[batch_indices]
            batch_size = len(batch_indices)
        else:
            data_batch = self.normalized_data
            batch_size = self.n_samples
        
        # Compute negative predictions: -y_i * <x_i, w>
        negative_predictions = -np.dot(data_batch, weights)
        
        # Compute quadratic coefficients: 0.5 * tanh(a/2) / a
        # Handle division by zero: when |a| is very small, use L'Hôpital's limit = 0.25
        quadratic_coefficients = np.divide(
            HALF * np.tanh(negative_predictions / 2),
            negative_predictions,
            out=np.ones(negative_predictions.shape) * QUARTER,
            where=np.abs(negative_predictions) > EPSILON
        )
        
        # Compute quadratic bound matrix
        weighted_data = data_batch.T * quadratic_coefficients
        quadratic_matrix = np.dot(weighted_data, data_batch)
        
        # Add regularization term
        regularization_term = self.regularization_strength * np.eye(self.n_features)
        
        return quadratic_matrix / batch_size + regularization_term
    
    def quadratic_upper_bound_without_regularization(
        self, 
        weights: np.ndarray, 
        batch_indices: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute quadratic upper bound without regularization term.
        
        Args:
            weights: Parameter vector of shape (n_features,)
            batch_indices: Optional indices for mini-batch computation
            
        Returns:
            Quadratic term matrix without regularization
        """
        # Select batch or use full data
        if batch_indices is not None:
            data_batch = self.normalized_data[batch_indices]
            batch_size = len(batch_indices)
        else:
            data_batch = self.normalized_data
            batch_size = self.n_samples
        
        # Compute negative predictions
        negative_predictions = -np.dot(data_batch, weights)
        
        # Compute quadratic coefficients with numerical stability
        quadratic_coefficients = np.divide(
            HALF * np.tanh(negative_predictions / 2),
            negative_predictions,
            out=np.ones(negative_predictions.shape) * QUARTER,
            where=np.abs(negative_predictions) > EPSILON
        )
        
        # Compute and return quadratic bound matrix
        weighted_data = data_batch.T * quadratic_coefficients
        quadratic_matrix = np.dot(weighted_data, data_batch)
        
        return quadratic_matrix / batch_size
    

    loss_wor = loss_without_regularization
    grad = gradient
    grad_wor = gradient_without_regularization
    hess = hessian
    hess_wor = hessian_without_regularization
    upperbound = quadratic_upper_bound
    upperbound_wor = quadratic_upper_bound_without_regularization
