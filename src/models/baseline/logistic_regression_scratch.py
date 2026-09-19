
from __future__ import annotations

import numpy as np


class LogisticRegressionScratch:
    

    def __init__(
        self,
        learning_rate: float = 0.5,
        epochs: int = 1000,
        threshold: float = 0.5,
        l2: float = 0.001,
        class_weight: str | None = "balanced",
        tolerance: float = 1e-8,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if not 0 < threshold < 1:
            raise ValueError("threshold must be between 0 and 1")
        if l2 < 0:
            raise ValueError("l2 must be non-negative")
        if class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or 'balanced'")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.threshold = float(threshold)
        self.l2 = float(l2)
        self.class_weight = class_weight
        self.tolerance = float(tolerance)

        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0
        self.classes_ = np.array([0, 1], dtype=np.int64)

    @staticmethod
    def _sigmoid(scores: np.ndarray) -> np.ndarray:
        """Return a numerically stable sigmoid."""
        clipped = np.clip(scores, -250.0, 250.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _sample_weights(self, targets: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones_like(targets, dtype=np.float64)

        counts = np.bincount(targets, minlength=2)
        if np.any(counts == 0):
            raise ValueError("balanced class weights require both classes")

        # n_samples / (n_classes * class_count)
        class_weights = len(targets) / (2.0 * counts)
        return class_weights[targets]

    def _binary_cross_entropy(
        self,
        targets: np.ndarray,
        probabilities: np.ndarray,
        sample_weights: np.ndarray,
    ) -> float:
        """Calculate weighted binary cross-entropy plus L2 penalty."""
        epsilon = 1e-12
        probabilities = np.clip(probabilities, epsilon, 1.0 - epsilon)
        losses = -(
            targets * np.log(probabilities)
            + (1.0 - targets) * np.log(1.0 - probabilities)
        )
        data_loss = float(np.sum(sample_weights * losses) / np.sum(sample_weights))
        regularization = 0.5 * self.l2 * float(np.dot(self.weights_, self.weights_))
        return data_loss + regularization

    def fit(self, features, targets) -> "LogisticRegressionScratch":
        """Learn weights and bias using explicit batch gradient descent."""
        if not hasattr(features, "shape") or len(features.shape) != 2:
            raise ValueError("features must be a two-dimensional matrix")

        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must contain the same samples")
        if features.shape[0] == 0:
            raise ValueError("training data cannot be empty")
        if not np.all(np.isin(targets, self.classes_)):
            raise ValueError("targets must contain only binary labels 0 and 1")

        self.weights_ = np.zeros(features.shape[1], dtype=np.float64)
        self.bias_ = 0.0
        self.loss_history_ = []
        self.n_iter_ = 0

        sample_weights = self._sample_weights(targets)
        normalizer = float(np.sum(sample_weights))
        previous_loss = np.inf

        for epoch in range(self.epochs):
            # Forward pass: z = Xw + b, followed by the sigmoid.
            scores = np.asarray(features @ self.weights_).reshape(-1) + self.bias_
            probabilities = self._sigmoid(scores)

            # Explicit binary cross-entropy for reporting and convergence.
            loss = self._binary_cross_entropy(
                targets,
                probabilities,
                sample_weights,
            )
            self.loss_history_.append(loss)

            # Explicit BCE gradients:
            # dw = X.T @ (p - y) / n; db = mean(p - y).
            weighted_error = sample_weights * (probabilities - targets)
            weight_gradient = np.asarray(
                features.T @ weighted_error
            ).reshape(-1) / normalizer
            weight_gradient += self.l2 * self.weights_
            bias_gradient = float(np.sum(weighted_error) / normalizer)

            # Explicit gradient-descent parameter update.
            self.weights_ -= self.learning_rate * weight_gradient
            self.bias_ -= self.learning_rate * bias_gradient
            self.n_iter_ = epoch + 1

            if self.tolerance and abs(previous_loss - loss) <= self.tolerance:
                break
            previous_loss = loss

        return self

    def _check_is_fitted(self) -> None:
        if self.weights_ is None:
            raise RuntimeError("the classifier must be fitted before prediction")

    def decision_function(self, features) -> np.ndarray:
        """Calculate the explicit linear score `Xw + b`."""
        self._check_is_fitted()
        if features.shape[1] != self.weights_.shape[0]:
            raise ValueError("feature count does not match the fitted model")
        return np.asarray(features @ self.weights_).reshape(-1) + self.bias_

    def predict_proba(self, features) -> np.ndarray:
        """Return SAFE and PII probabilities for each record."""
        pii_probability = self._sigmoid(self.decision_function(features))
        return np.column_stack((1.0 - pii_probability, pii_probability))

    def predict(self, features) -> np.ndarray:
        """Return binary predictions using the configured threshold."""
        return (self.predict_proba(features)[:, 1] >= self.threshold).astype(np.int64)

    def score(self, features, targets) -> float:
        """Return classification accuracy."""
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        return float(np.mean(self.predict(features) == targets))
