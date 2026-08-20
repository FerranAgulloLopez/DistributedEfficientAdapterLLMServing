import numpy as np
import numba as nb


def export_sklearn_tree_classifier(tree, classes_):
    feature = np.asarray(tree.feature, dtype=np.int32)
    threshold = np.asarray(tree.threshold, dtype=np.float64)
    left = np.asarray(tree.children_left, dtype=np.int32)
    right = np.asarray(tree.children_right, dtype=np.int32)

    # value: (n_nodes, 1, n_classes)
    value = np.asarray(tree.value, dtype=np.float64)[:, 0, :]
    sums = value.sum(axis=1, keepdims=True)
    leaf_proba = value / np.where(sums == 0.0, 1.0, sums)

    classes_ = np.asarray(classes_)  # keep outside numba for label mapping
    return feature, threshold, left, right, leaf_proba, classes_


class FastTreeClassifierNumba:
    def __init__(self, tree, classes_):
        (self.feature,
         self.threshold,
         self.left,
         self.right,
         self.leaf_proba,
         self.classes_) = export_sklearn_tree_classifier(tree, classes_)

        # Warm-up compile (optional but recommended to avoid first-call latency later)
        X0 = np.zeros((1, int(self.feature[self.feature >= 0].max() + 1)) if np.any(self.feature >= 0) else 1,
                      dtype=np.float64)
        _ = predict_class_index_numba(X0, self.feature, self.threshold, self.left, self.right, self.leaf_proba)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        return predict_proba_numba(X, self.feature, self.threshold, self.left, self.right, self.leaf_proba)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        idx = predict_class_index_numba(X, self.feature, self.threshold, self.left, self.right, self.leaf_proba)
        return self.classes_[idx]


@nb.njit(cache=True)
def _apply_one_row(x_row, feature, threshold, left, right):
    node = 0
    while feature[node] != -2:  # -2 denotes leaf in sklearn trees
        f = feature[node]
        v = x_row[f]
        thr = threshold[node]
        if v > thr:
            node = right[node]
        else:
            node = left[node]
    return node


@nb.njit(cache=True)
def apply_numba(X, feature, threshold, left, right):
    n = X.shape[0]
    out = np.empty(n, dtype=np.int32)
    for i in range(n):
        out[i] = _apply_one_row(X[i], feature, threshold, left, right)
    return out


@nb.njit(cache=True)
def predict_proba_numba(X, feature, threshold, left, right, leaf_proba):
    n = X.shape[0]
    n_classes = leaf_proba.shape[1]
    out = np.empty((n, n_classes), dtype=np.float64)

    for i in range(n):
        leaf = _apply_one_row(X[i], feature, threshold, left, right)
        # copy proba row
        for c in range(n_classes):
            out[i, c] = leaf_proba[leaf, c]
    return out


@nb.njit(cache=True)
def predict_class_index_numba(X, feature, threshold, left, right, leaf_proba):
    n = X.shape[0]
    out = np.empty(n, dtype=np.int32)

    for i in range(n):
        leaf = _apply_one_row(X[i], feature, threshold, left, right)
        # argmax
        best_c = 0
        best_v = leaf_proba[leaf, 0]
        for c in range(1, leaf_proba.shape[1]):
            v = leaf_proba[leaf, c]
            if v > best_v:
                best_v = v
                best_c = c
        out[i] = best_c
    return out


@nb.njit(cache=True)
def _apply_one_row_reg(x_row, feature, threshold, left, right):
    node = 0
    while feature[node] != -2:
        f = feature[node]
        v = x_row[f]
        thr = threshold[node]
        if v > thr:
            node = right[node]
        else:
            node = left[node]
    return node

@nb.njit(cache=True)
def predict_one_regression_numba_scalar(x_row, feature, threshold, left, right, leaf_value):
    leaf = _apply_one_row_reg(x_row, feature, threshold, left, right)
    return leaf_value[leaf, 0]

@nb.njit(cache=True)
def predict_one_regression_numba_vector(x_row, feature, threshold, left, right, leaf_value):
    leaf = _apply_one_row_reg(x_row, feature, threshold, left, right)
    n_outputs = leaf_value.shape[1]
    out = np.empty(n_outputs, dtype=np.float64)
    for j in range(n_outputs):
        out[j] = leaf_value[leaf, j]
    return out

@nb.njit(cache=True)
def predict_regression_numba_scalar(X, feature, threshold, left, right, leaf_value):
    n = X.shape[0]
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        leaf = _apply_one_row_reg(X[i], feature, threshold, left, right)
        out[i] = leaf_value[leaf, 0]
    return out

@nb.njit(cache=True)
def predict_regression_numba_vector(X, feature, threshold, left, right, leaf_value):
    n = X.shape[0]
    n_outputs = leaf_value.shape[1]
    out = np.empty((n, n_outputs), dtype=np.float64)
    for i in range(n):
        leaf = _apply_one_row_reg(X[i], feature, threshold, left, right)
        for j in range(n_outputs):
            out[i, j] = leaf_value[leaf, j]
    return out

def export_sklearn_tree_regressor(tree):
    feature = np.asarray(tree.feature, dtype=np.int32)
    threshold = np.asarray(tree.threshold, dtype=np.float64)
    left = np.asarray(tree.children_left, dtype=np.int32)
    right = np.asarray(tree.children_right, dtype=np.int32)

    value = np.asarray(tree.value, dtype=np.float64)
    if value.ndim != 3 or value.shape[2] != 1:
        raise ValueError(f"Unexpected tree.value shape for regressor: {value.shape}")
    leaf_value = value[:, :, 0].copy(order="C")  # (n_nodes, n_outputs)

    return feature, threshold, left, right, leaf_value

class FastTreeRegressorNumba:
    def __init__(self, tree):
        self.feature, self.threshold, self.left, self.right, self.leaf_value = export_sklearn_tree_regressor(tree)
        self.n_outputs_ = int(self.leaf_value.shape[1])

        # Warm-up compile (optional)
        max_f = -1
        for i in range(self.feature.shape[0]):
            if self.feature[i] > max_f:
                max_f = self.feature[i]
        n_feat = max_f + 1 if max_f >= 0 else 1
        x0 = np.zeros(n_feat, dtype=np.float64)

        if self.n_outputs_ == 1:
            _ = predict_one_regression_numba_scalar(
                x0, self.feature, self.threshold, self.left, self.right, self.leaf_value
            )
        else:
            _ = predict_one_regression_numba_vector(
                x0, self.feature, self.threshold, self.left, self.right, self.leaf_value
            )

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            if self.n_outputs_ == 1:
                return float(predict_one_regression_numba_scalar(
                    X, self.feature, self.threshold, self.left, self.right, self.leaf_value
                ))
            return predict_one_regression_numba_vector(
                X, self.feature, self.threshold, self.left, self.right, self.leaf_value
            )

        if self.n_outputs_ == 1:
            return predict_regression_numba_scalar(
                X, self.feature, self.threshold, self.left, self.right, self.leaf_value
            )
        return predict_regression_numba_vector(
            X, self.feature, self.threshold, self.left, self.right, self.leaf_value
        )
