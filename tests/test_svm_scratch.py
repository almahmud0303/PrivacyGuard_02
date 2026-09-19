import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.baseline.svm_scratch import LinearSVMScratch


def test_scratch_svm_learns_sparse_linearly_separable_data():
    features = csr_matrix(
        [
            [-3.0, -1.0],
            [-2.0, -1.0],
            [-1.0, -2.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 1.0],
        ]
    )
    targets = np.array([0, 0, 0, 1, 1, 1])
    model = LinearSVMScratch(
        learning_rate=0.5,
        epochs=1000,
        l2=0.01,
        class_weight=None,
    ).fit(features, targets)

    assert np.array_equal(model.predict(features), targets)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_scratch_svm_exposes_linear_score():
    features = csr_matrix([[-2.0], [-1.0], [1.0], [2.0]])
    targets = np.array([0, 0, 1, 1])
    model = LinearSVMScratch(epochs=500).fit(features, targets)

    expected = np.asarray(features @ model.weights_).reshape(-1) + model.bias_
    assert np.allclose(model.decision_function(features), expected)


def test_scratch_svm_rejects_non_binary_targets():
    model = LinearSVMScratch()
    with pytest.raises(ValueError, match="binary labels"):
        model.fit(csr_matrix([[0.0], [1.0], [2.0]]), np.array([0, 1, 2]))
