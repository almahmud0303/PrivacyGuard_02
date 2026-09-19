"""Linear binary SVM implemented with NumPy subgradient descent."""
from __future__ import annotations

import numpy as np


class LinearSVMScratch:
    """Linear SVM using hinge loss, L2 regularization, and batch updates."""

    def __init__(
        self,
        learning_rate: float = 0.5,
        epochs: int = 2000,
        l2: float = 0.01,
        class_weight: str | None = "balanced",
        learning_rate_decay: bool = True,
        tolerance: float = 0.0,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if l2 <= 0:
            raise ValueError("l2 must be positive")
        if class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or 'balanced'")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.l2 = float(l2)
        self.class_weight = class_weight
        self.learning_rate_decay = bool(learning_rate_decay)
        self.tolerance = float(tolerance)
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0
        self.classes_ = np.array([0, 1], dtype=np.int64)

    def _sample_weights(self, targets: np.ndarray) -> np.ndarray:
        if self.class_weight is None:
            return np.ones_like(targets, dtype=np.float64)
        counts = np.bincount(targets, minlength=2)
        if np.any(counts == 0):
            raise ValueError("balanced class weights require both classes")
        class_weights = len(targets) / (2.0 * counts)
        return class_weights[targets]

    def fit(self, features, targets) -> "LinearSVMScratch":
        """Optimize the primal linear SVM objective."""
        if not hasattr(features, "shape") or len(features.shape) != 2:
            raise ValueError("features must be a two-dimensional matrix")
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must contain the same samples")
        if features.shape[0] == 0:
            raise ValueError("training data cannot be empty")
        if not np.all(np.isin(targets, self.classes_)):
            raise ValueError("targets must contain only binary labels 0 and 1")

        signed_targets = np.where(targets == 1, 1.0, -1.0)
        sample_weights = self._sample_weights(targets)
        normalizer = float(sample_weights.sum())
        self.weights_ = np.zeros(features.shape[1], dtype=np.float64)
        self.bias_ = 0.0
        self.loss_history_ = []
        self.n_iter_ = 0
        previous_loss = np.inf

        for epoch in range(self.epochs):
            scores = np.asarray(features @ self.weights_).reshape(-1) + self.bias_
            margins = 1.0 - signed_targets * scores
            hinge_losses = np.maximum(0.0, margins)
            loss = (
                0.5 * self.l2 * float(np.dot(self.weights_, self.weights_))
                + float(np.sum(sample_weights * hinge_losses) / normalizer)
            )
            self.loss_history_.append(loss)

            active = margins > 0
            weight_gradient = self.l2 * self.weights_
            bias_gradient = 0.0
            if np.any(active):
                active_scale = sample_weights[active] * signed_targets[active]
                weight_gradient -= np.asarray(
                    features[active].T @ active_scale
                ).reshape(-1) / normalizer
                bias_gradient = -float(np.sum(active_scale) / normalizer)

            step = self.learning_rate
            if self.learning_rate_decay:
                step /= np.sqrt(epoch + 1.0)
            self.weights_ -= step * weight_gradient
            self.bias_ -= step * bias_gradient
            self.n_iter_ = epoch + 1

            if self.tolerance and abs(previous_loss - loss) <= self.tolerance:
                break
            previous_loss = loss
        return self

    def _check_is_fitted(self) -> None:
        if self.weights_ is None:
            raise RuntimeError("the classifier must be fitted before prediction")

    def decision_function(self, features) -> np.ndarray:
        """Calculate the signed distance score Xw + b."""
        self._check_is_fitted()
        if features.shape[1] != self.weights_.shape[0]:
            raise ValueError("feature count does not match the fitted model")
        return np.asarray(features @ self.weights_).reshape(-1) + self.bias_

    def predict(self, features) -> np.ndarray:
        return (self.decision_function(features) >= 0.0).astype(np.int64)

    def score(self, features, targets) -> float:
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        return float(np.mean(self.predict(features) == targets))
