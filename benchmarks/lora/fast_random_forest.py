import numpy as np

class FastSklearnTreeClassifier:
    """
    Fast evaluator for a fitted sklearn DecisionTreeClassifier tree_.
    Works also for RandomForestClassifier with one estimator by passing estimators_[0].tree_.
    """
    def __init__(self, tree, classes_):
        self.classes_ = np.asarray(classes_)
        self.n_classes_ = len(self.classes_)

        # Copy into contiguous numpy arrays for speed
        self.feature = np.asarray(tree.feature, dtype=np.int32)
        self.threshold = np.asarray(tree.threshold, dtype=np.float64)
        self.left = np.asarray(tree.children_left, dtype=np.int32)
        self.right = np.asarray(tree.children_right, dtype=np.int32)

        # value shape: (n_nodes, 1, n_classes) for classifier
        value = np.asarray(tree.value, dtype=np.float64)
        self.leaf_class_counts = value[:, 0, :]  # (n_nodes, n_classes)

        # Precompute leaf probabilities (avoid division during predict_proba)
        sums = self.leaf_class_counts.sum(axis=1, keepdims=True)
        # For safety (shouldn't be zero at leaves, but anyway):
        self.leaf_proba = np.divide(
            self.leaf_class_counts,
            np.where(sums == 0.0, 1.0, sums),
        )

    def _apply_one(self, x_row):
        """Return leaf node id for one row."""
        node = 0
        while self.feature[node] != -2:  # internal node
            f = self.feature[node]
            thr = self.threshold[node]
            v = x_row[f]

            if v > thr:
                node = self.right[node]
            else:
                node = self.left[node]
        return node

    def apply(self, X):
        """Return leaf node ids for each row."""
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        out = np.empty(n, dtype=np.int32)
        for i in range(n):
            out[i] = self._apply_one(X[i])
        return out

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        leaves = self.apply(X)
        return self.leaf_proba[leaves]

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]

class FastSklearnTreeRegressor:
    """
    Fast pure-Python evaluator for a fitted sklearn DecisionTreeRegressor.tree_.
    Also works for RandomForestRegressor with one estimator by passing estimators_[0].tree_.

    Supports single-output and multi-output regression.
    """
    def __init__(self, tree):
        self.feature = np.asarray(tree.feature, dtype=np.int32)
        self.threshold = np.asarray(tree.threshold, dtype=np.float64)
        self.left = np.asarray(tree.children_left, dtype=np.int32)
        self.right = np.asarray(tree.children_right, dtype=np.int32)

        # value shape for regressor: (n_nodes, n_outputs, 1) in sklearn
        value = np.asarray(tree.value, dtype=np.float64)
        if value.ndim != 3 or value.shape[2] != 1:
            raise ValueError(f"Unexpected tree.value shape for regressor: {value.shape}")
        self.leaf_value = value[:, :, 0]  # (n_nodes, n_outputs)
        self.n_outputs_ = self.leaf_value.shape[1]

    def _apply_one(self, x_row):
        node = 0
        while self.feature[node] != -2:  # internal node
            f = self.feature[node]
            v = x_row[f]
            thr = self.threshold[node]
            if v > thr:
                node = self.right[node]
            else:
                node = self.left[node]
        return node

    def apply(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            return self._apply_one(X)
        n = X.shape[0]
        out = np.empty(n, dtype=np.int32)
        for i in range(n):
            out[i] = self._apply_one(X[i])
        return out

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            leaf = self._apply_one(X)
            y = self.leaf_value[leaf]
            return float(y[0]) if self.n_outputs_ == 1 else y.copy()

        leaves = self.apply(X)
        y = self.leaf_value[leaves]  # (n_samples, n_outputs)
        if self.n_outputs_ == 1:
            return y[:, 0].copy()
        return y.copy()


class FastSklearnTreeRegressor:
    """
    Fast pure-Python evaluator for a fitted sklearn DecisionTreeRegressor.tree_.
    Also works for RandomForestRegressor with one estimator by passing estimators_[0].tree_.

    Supports single-output and multi-output regression.
    """
    def __init__(self, tree):
        self.feature = np.asarray(tree.feature, dtype=np.int32)
        self.threshold = np.asarray(tree.threshold, dtype=np.float64)
        self.left = np.asarray(tree.children_left, dtype=np.int32)
        self.right = np.asarray(tree.children_right, dtype=np.int32)

        # value shape for regressor: (n_nodes, n_outputs, 1) in sklearn
        value = np.asarray(tree.value, dtype=np.float64)
        if value.ndim != 3 or value.shape[2] != 1:
            raise ValueError(f"Unexpected tree.value shape for regressor: {value.shape}")
        self.leaf_value = value[:, :, 0]  # (n_nodes, n_outputs)
        self.n_outputs_ = self.leaf_value.shape[1]

    def _apply_one(self, x_row):
        node = 0
        while self.feature[node] != -2:  # internal node
            f = self.feature[node]
            v = x_row[f]
            thr = self.threshold[node]
            if v > thr:
                node = self.right[node]
            else:
                node = self.left[node]
        return node

    def apply(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            return self._apply_one(X)
        n = X.shape[0]
        out = np.empty(n, dtype=np.int32)
        for i in range(n):
            out[i] = self._apply_one(X[i])
        return out

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            leaf = self._apply_one(X)
            y = self.leaf_value[leaf]
            return float(y[0]) if self.n_outputs_ == 1 else y.copy()

        leaves = self.apply(X)
        y = self.leaf_value[leaves]  # (n_samples, n_outputs)
        if self.n_outputs_ == 1:
            return y[:, 0].copy()
        return y.copy()
