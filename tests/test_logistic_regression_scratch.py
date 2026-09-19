import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.baseline.logistic_regression_scratch import (
    LogisticRegressionScratch,
)


def test_scratch_logistic_regression_learns_sparse_binary_data():
    features = csr_matrix(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ]
    )
    targets = np.array([0, 0, 0, 1, 1, 1])

    model = LogisticRegressionScratch(
        learning_rate=0.5,
        epochs=2000,
        l2=0.0,
        class_weight=None,
        tolerance=0.0,
    )
    model.fit(features, targets)

    assert np.array_equal(model.predict(features), targets)
    assert model.loss_history_[-1] < model.loss_history_[0]
    assert model.predict_proba(features).shape == (len(targets), 2)
    assert np.allclose(model.predict_proba(features).sum(axis=1), 1.0)


def test_scratch_logistic_regression_exposes_linear_score():
    features = csr_matrix([[0.0], [1.0], [2.0], [3.0]])
    targets = np.array([0, 0, 1, 1])
    model = LogisticRegressionScratch(epochs=1000, tolerance=0.0)
    model.fit(features, targets)

    expected = np.asarray(features @ model.weights_).reshape(-1) + model.bias_
    assert np.allclose(model.decision_function(features), expected)


def test_scratch_logistic_regression_rejects_non_binary_targets():
    model = LogisticRegressionScratch()
    with pytest.raises(ValueError, match="binary labels"):
        model.fit(csr_matrix([[0.0], [1.0], [2.0]]), np.array([0, 1, 2]))
