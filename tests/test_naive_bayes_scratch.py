import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.baseline.naive_bayes_scratch import (
    MultinomialNaiveBayesScratch,
)


def test_scratch_naive_bayes_learns_sparse_nonnegative_data():
    features = csr_matrix(
        [
            [4.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [0.0, 1.0, 3.0],
            [0.0, 0.0, 4.0],
        ]
    )
    targets = np.array([0, 0, 1, 1])
    model = MultinomialNaiveBayesScratch(alpha=1.0).fit(features, targets)

    assert np.array_equal(model.predict(features), targets)
    assert model.predict_proba(features).shape == (4, 2)
    assert np.allclose(model.predict_proba(features).sum(axis=1), 1.0)
    assert np.all(model.feature_count_ >= 0)


def test_scratch_naive_bayes_uses_laplace_smoothing():
    features = csr_matrix([[2.0, 0.0], [0.0, 2.0]])
    targets = np.array([0, 1])
    model = MultinomialNaiveBayesScratch(alpha=1.0).fit(features, targets)

    assert np.all(np.isfinite(model.feature_log_prob_))


def test_scratch_naive_bayes_rejects_negative_features():
    model = MultinomialNaiveBayesScratch()
    with pytest.raises(ValueError, match="non-negative"):
        model.fit(csr_matrix([[-1.0], [1.0]]), np.array([0, 1]))
