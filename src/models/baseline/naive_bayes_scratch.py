"""Multinomial Naive Bayes implemented directly with NumPy."""
from __future__ import annotations

import numpy as np


class MultinomialNaiveBayesScratch:
    """Binary Multinomial Naive Bayes with Laplace smoothing."""

    def __init__(self, alpha: float = 1.0, fit_prior: bool = True) -> None:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = float(alpha)
        self.fit_prior = bool(fit_prior)
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.class_count_: np.ndarray | None = None
        self.feature_count_: np.ndarray | None = None
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None

    @staticmethod
    def _validate_features(features) -> None:
        if not hasattr(features, "shape") or len(features.shape) != 2:
            raise ValueError("features must be a two-dimensional matrix")
        if features.shape[0] == 0:
            raise ValueError("training data cannot be empty")
        if float(features.min()) < 0:
            raise ValueError("Multinomial Naive Bayes requires non-negative features")

    def fit(self, features, targets) -> "MultinomialNaiveBayesScratch":
        """Estimate priors and per-class feature likelihoods."""
        self._validate_features(features)
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        if features.shape[0] != targets.shape[0]:
            raise ValueError("features and targets must contain the same samples")
        if not np.all(np.isin(targets, self.classes_)):
            raise ValueError("targets must contain only binary labels 0 and 1")

        self.class_count_ = np.bincount(targets, minlength=2).astype(np.float64)
        if np.any(self.class_count_ == 0):
            raise ValueError("training requires examples from both classes")

        self.feature_count_ = np.zeros((2, features.shape[1]), dtype=np.float64)
        for class_id in self.classes_:
            class_rows = features[targets == class_id]
            self.feature_count_[class_id] = np.asarray(
                class_rows.sum(axis=0)
            ).reshape(-1)

        # Laplace smoothing: count(feature, class) + alpha.
        smoothed_counts = self.feature_count_ + self.alpha
        smoothed_totals = smoothed_counts.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_counts / smoothed_totals)

        if self.fit_prior:
            self.class_log_prior_ = np.log(
                self.class_count_ / self.class_count_.sum()
            )
        else:
            self.class_log_prior_ = np.full(2, -np.log(2.0))
        return self

    def _check_is_fitted(self) -> None:
        if self.feature_log_prob_ is None or self.class_log_prior_ is None:
            raise RuntimeError("the classifier must be fitted before prediction")

    def joint_log_likelihood(self, features) -> np.ndarray:
        """Calculate log P(class) plus the feature log-likelihood sum."""
        self._check_is_fitted()
        if features.shape[1] != self.feature_log_prob_.shape[1]:
            raise ValueError("feature count does not match the fitted model")
        if float(features.min()) < 0:
            raise ValueError("Multinomial Naive Bayes requires non-negative features")
        return (
            np.asarray(features @ self.feature_log_prob_.T)
            + self.class_log_prior_
        )

    def predict_log_proba(self, features) -> np.ndarray:
        scores = self.joint_log_likelihood(features)
        maximum = scores.max(axis=1, keepdims=True)
        log_normalizer = maximum + np.log(
            np.exp(scores - maximum).sum(axis=1, keepdims=True)
        )
        return scores - log_normalizer

    def predict_proba(self, features) -> np.ndarray:
        return np.exp(self.predict_log_proba(features))

    def predict(self, features) -> np.ndarray:
        scores = self.joint_log_likelihood(features)
        return self.classes_[np.argmax(scores, axis=1)]

    def score(self, features, targets) -> float:
        targets = np.asarray(targets, dtype=np.int64).reshape(-1)
        return float(np.mean(self.predict(features) == targets))
