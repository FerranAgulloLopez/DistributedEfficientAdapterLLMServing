import argparse
import csv
import json
import os
import random
import time
from typing import List, Tuple, Dict, Any
from functools import partial
from sklearn.tree import export_graphviz
import graphviz
import re

import joblib
import numpy as np
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.tree import _tree
from sklearn.metrics import accuracy_score, f1_score, fbeta_score
from fast_random_forest import FastSklearnTreeClassifier, FastSklearnTreeRegressor
from fast_random_forest_numba import FastTreeClassifierNumba, FastTreeRegressorNumba

ALL_X_FEATURES: List[str] = ['sum_rate', 'std_rate', 'max_size', 'mean_size', 'std_size', 'adapter_slots', 'served_adapters']
ALL_Y_FEATURES: List[str] = ['total_throughput', 'starvation', 'itl', 'ttft']
REG_Y_FEATURES: List[str] = ['total_throughput', 'itl', 'ttft']
CLASS_Y_FEATURES: List[str] = ['starvation']


def collapse_equal_leaves(tree):
    """
    Collapse internal nodes if both children are leaves
    and predict the same value.
    """
    def _prune_node(node_id):
        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        # If it's already a leaf, nothing to do
        if left == -1 and right == -1:
            return

        # Recurse on children first
        if left != -1:
            _prune_node(left)
        if right != -1:
            _prune_node(right)

        # After recursion, check if both children are leaves
        if (tree.children_left[left] == -1 and
            tree.children_left[right] == -1):

            # Compare predictions
            left_val = tree.value[left][0,0]
            right_val = tree.value[right][0,0]

            if np.isclose(left_val, right_val):
                # Make current node a leaf
                tree.children_left[node_id] = -1
                tree.children_right[node_id] = -1
                tree.feature[node_id] = -2   # convention for "no split"
                tree.threshold[node_id] = -2.0
                tree.value[node_id][0,0] = left_val  # keep that prediction

    _prune_node(0)  # start at root


def tree_to_rules(tree, feature_names):
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree.feature
    ]

    paths = []

    def recurse(node, path):
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree.threshold[node]
            # Left child
            recurse(tree.children_left[node], path + [f"{name} <= {threshold:.3f}"])
            # Right child
            recurse(tree.children_right[node], path + [f"{name} > {threshold:.3f}"])
        else:
            # Leaf node
            paths.append((path, tree.value[node][0][0]))

    recurse(0, [])
    return paths


def find_tree_n_rules(feature_names, estimator):
    # extract single tree from random forest
    tree = estimator.estimators_[0].tree_

    # collapse leaves
    collapse_equal_leaves(tree)

    # extract number of rules
    rules = tree_to_rules(tree, feature_names)

    return len(rules)


def smape(y_true: List[float], y_pred: List[float]):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred))
    diff = np.abs(y_true - y_pred)
    output = float(100 * np.mean(diff / denominator))
    return output


class ScoringWithRulesNumber:

    def __init__(
            self,
            feature_names: List[str],
            scorer_function,
            max_rules: int,
            greater_is_better: bool,
    ):
        self.feature_names = feature_names
        self.scorer_function = scorer_function
        self.max_rules = max_rules
        self.greater_is_better = greater_is_better

    def score(self, estimator, X, y_true):
        n_rules: int = find_tree_n_rules(feature_names=self.feature_names, estimator=estimator)
        if n_rules > self.max_rules:
            if self.greater_is_better:
                return -1e9
            else:
                return 1e9
        y_pred = estimator.predict(X)
        output = self.scorer_function(y_true, y_pred)
        if not self.greater_is_better:
            return -output
        return output


def load_dataset(path: str, x_features, y_features) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    dataset_x: List[List[Any]] = []
    dataset_y: List[List[Any]] = []
    paths: List[str] = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row_x = []
            for feature in x_features:
                row_x.append(float(row[feature]))
            dataset_x.append(row_x)
            row_y = []
            for feature in y_features:
                row_y.append(float(row[feature]))
            dataset_y.append(row_y)
            paths.append(row['path'])
    dataset_x: np.ndarray = np.asarray(dataset_x)
    dataset_y: np.ndarray = np.asarray(dataset_y)
    return dataset_x, dataset_y, paths


def train_model(
        X_train,
        y_train,
        X_test,
        y_test,
        random_seed: int,
        output_path: str,
        import_statement: str,
        class_name: str,
        parameters_to_test: dict,
        y_features: List[str],
        classification: bool,
        fbeta_value: bool,
        max_rules: int,
        feature_names: List[str],
) -> None:
    # define model
    model_name = os.path.basename(output_path)
    exec(import_statement)
    model = eval(class_name)()

    # only implement for RandomForest with one estimator max
    if not classification:
        if class_name != 'RandomForestRegressor':
            raise ValueError('Explainability only implemented for sklearn RandomForestRegressor in regression scenario')
    else:
        if class_name != 'RandomForestClassifier':
            raise ValueError('Explainability only implemented for sklearn RandomForestClassifier in classification scenario')
    if 'n_estimators' not in parameters_to_test:
        raise ValueError('Explainability only implemented for n_estimators=1')
    if 'n_estimators' in parameters_to_test and any([value != 1 for value in parameters_to_test['n_estimators']]):
        raise ValueError('Explainability only implemented for n_estimators=1')

    # prepare csv header
    if not classification:
        row = ['target', 'model', 'r_2', 'smape', 'time']
    else:
        row = ['target', 'model', 'accuracy', 'f1_macro', 'time']
    with open(os.path.join(output_path, f'test_results.csv'), mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(row)

    # prepare scorer (taking into account also maximum number of rules)
    if not classification:
        score_class = ScoringWithRulesNumber(
            scorer_function=smape,
            max_rules=max_rules,
            feature_names=feature_names,
            greater_is_better=False
        )
        scoring = score_class.score
    else:
        if fbeta_value == 1:
            scorer_function = partial(f1_score, average='macro')
        else:
            scorer_function = partial(fbeta_score, beta=fbeta_value, average='macro')
        score_class = ScoringWithRulesNumber(
            scorer_function=scorer_function,
            max_rules=max_rules,
            feature_names=feature_names,
            greater_is_better=True
        )
        scoring = score_class.score

    # train and test model
    for y_feature_index, y_feature_label in enumerate(y_features):
        print(f'\n\nTraining to predict {y_feature_label}')

        # train model and find the best hyperparameters with cv
        estimator = HalvingGridSearchCV(
            estimator=model,
            param_grid=parameters_to_test,
            cv=5,
            scoring=scoring,
            n_jobs=-1
        )
        estimator.fit(X_train, y_train[:, y_feature_index])

        # create fast version
        if classification:
            fast = FastSklearnTreeClassifier(
                estimator.best_estimator_.estimators_[0].tree_,
                estimator.best_estimator_.classes_
            )
        else:
            fast = FastSklearnTreeRegressor(
                estimator.best_estimator_.estimators_[0].tree_,
            )
        init_time: float = time.perf_counter()
        y_pred_test = fast.predict(X_test)
        prediction_time: float = ((time.perf_counter() - init_time) / len(X_test)) * 1000
        if not classification:
            r2_result = r2_score(y_test[:, y_feature_index], y_pred_test)
            smape_result = smape(y_test[:, y_feature_index], y_pred_test)
            print(f'FAST. Training finished. Obtained results in test: R2 -> {r2_result}; SMAPE -> {smape_result}; time -> {prediction_time}')
        else:
            accuracy = accuracy_score(y_test[:, y_feature_index], y_pred_test)
            f1_macro = f1_score(y_test[:, y_feature_index], y_pred_test, average='macro')
            print(f'FAST. Training finished. Obtained results in test: Accuracy -> {accuracy}; F1-macro -> {f1_macro}; time -> {prediction_time}')

        # create fast version with NUMBA
        if classification:
            fast = FastTreeClassifierNumba(
                estimator.best_estimator_.estimators_[0].tree_,
                estimator.best_estimator_.classes_
            )
        else:
            fast = FastTreeRegressorNumba(
                estimator.best_estimator_.estimators_[0].tree_,
            )
        _ = fast.predict(np.empty((2, np.shape(X_test)[1])))  # compile first
        init_time: float = time.perf_counter()
        y_pred_test = fast.predict(X_test)
        prediction_time: float = ((time.perf_counter() - init_time) / len(X_test)) * 1000
        if not classification:
            r2_result = r2_score(y_test[:, y_feature_index], y_pred_test)
            smape_result = smape(y_test[:, y_feature_index], y_pred_test)
            print(f'FAST NUMBA. Training finished. Obtained results in test: R2 -> {r2_result}; SMAPE -> {smape_result}; time -> {prediction_time}')
        else:
            accuracy = accuracy_score(y_test[:, y_feature_index], y_pred_test)
            f1_macro = f1_score(y_test[:, y_feature_index], y_pred_test, average='macro')
            print(f'FAST NUMBA. Training finished. Obtained results in test: Accuracy -> {accuracy}; F1-macro -> {f1_macro}; time -> {prediction_time}')

        # extract test results of best model
        init_time: float = time.perf_counter()
        y_pred_test = estimator.best_estimator_.predict(X_test)
        prediction_time: float = ((time.perf_counter() - init_time) / len(X_test)) * 1000
        if not classification:
            r2_result = r2_score(y_test[:, y_feature_index], y_pred_test)
            smape_result = smape(y_test[:, y_feature_index], y_pred_test)
            print(f'Training finished. Obtained results in test: R2 -> {r2_result}; SMAPE -> {smape_result}; time -> {prediction_time}')
        else:
            accuracy = accuracy_score(y_test[:, y_feature_index], y_pred_test)
            f1_macro = f1_score(y_test[:, y_feature_index], y_pred_test, average='macro')
            print(f'Training finished. Obtained results in test: Accuracy -> {accuracy}; F1-macro -> {f1_macro}; time -> {prediction_time}')

        # save test results to csv
        if not classification:
            row = [
                y_feature_label,
                model_name,
                f'{r2_result:.2f}',
                f'{smape_result:.2f}',
                f'{prediction_time:.2f}',
            ]
        else:
            row = [
                y_feature_label,
                model_name,
                f'{accuracy:.2f}',
                f'{f1_macro:.2f}',
                f'{prediction_time:.2f}',
            ]
        with open(os.path.join(output_path, f'test_results.csv'), mode='a') as file:
            writer = csv.writer(file)
            writer.writerow(row)

        # save best model
        joblib.dump(estimator.best_estimator_, os.path.join(output_path, f'best_model_{y_feature_label}.pkl'))
        joblib.dump(fast, os.path.join(output_path, f'best_model_{y_feature_label}_fast.pkl'))

        # save best model params
        with open(os.path.join(output_path, f'best_params_{y_feature_label}.json'), 'w') as file:
            json.dump(estimator.best_params_, file, indent=4)

        # print and save rules
        tree = estimator.best_estimator_.estimators_[0].tree_
        rules = tree_to_rules(tree, feature_names)
        print(f'\n\n\n Rules for {y_feature_label} N# {len(rules)}')
        for path, value in rules:
            print(' AND '.join(path), '=>', value)

        # store tree visualization
        tree = estimator.best_estimator_.estimators_[0]
        dot_data = export_graphviz(
            tree,
            out_file=None,
            feature_names=feature_names,
            filled=True,
            rounded=True,
            special_characters=True,
            impurity=False,
            proportion=True,
            node_ids=False
        )
        new_lines = []
        for line in dot_data.splitlines():
            if 'label=<' in line:
                label_content = re.search(r'label=<(.*)>,', line).group(1)

                if label_content.startswith('samples = '):
                    # This is a leaf
                    new_lines.append(line)
                else:
                    # This is an internal node
                    cleaned = re.sub(r'<br\/>samples = (.*)', '', label_content)
                    new_line = re.sub(r'label=<(.*)>,', f'label=<{cleaned}>,', line)
                    new_lines.append(new_line)
            else:
                new_lines.append(line)

        dot_data = "\n".join(new_lines)

        def round_floats(match):
            return str(round(float(match.group()), 2))

        dot_data = re.sub(r'\d+\.\d+', round_floats, dot_data)
        graph = graphviz.Source(dot_data)
        os.makedirs(os.path.join(output_path, 'trees'))
        graph.render(os.path.join(output_path, 'trees', f'tree_{y_feature_label}'), format="png", cleanup=True)


def main(
        output_path: str,
        train_dataset_path: str,
        test_dataset_path: str,
        import_statement: str,
        class_name: str,
        parameters_to_test: dict,
        predict_classification_features: bool,
        fbeta_value: bool,
        max_rules: int,
):
    global ALL_X_FEATURES, REG_Y_FEATURES, CLASS_Y_FEATURES

    # set random seed
    random_seed: int = 0
    random.seed(random_seed)
    np.random.seed(random_seed)

    # load datasets
    if predict_classification_features:
        y_features = CLASS_Y_FEATURES
    else:
        y_features = REG_Y_FEATURES
    train_dataset_x, train_dataset_y, _ = load_dataset(
        train_dataset_path,
        ALL_X_FEATURES,
        y_features
    )
    test_dataset_x, test_dataset_y, _ = load_dataset(
        test_dataset_path,
        ALL_X_FEATURES,
        y_features
    )

    # transform to numpy
    X_train = np.asarray(train_dataset_x)
    y_train = np.asarray(train_dataset_y)
    X_test = np.asarray(test_dataset_x)
    y_test = np.asarray(test_dataset_y)

    # run training
    train_model(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_seed=random_seed,
        output_path=output_path,
        import_statement=import_statement,
        class_name=class_name,
        parameters_to_test=parameters_to_test,
        y_features=y_features,
        classification=predict_classification_features,
        fbeta_value=fbeta_value,
        max_rules=max_rules,
        feature_names=ALL_X_FEATURES,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Launcher for training ML models to replicate Digital Twin results')
    parser.add_argument('--output-path', type=str, help='Directory to store results', required=True)
    parser.add_argument('--train-dataset-path', type=str, help='Training dataset to use', required=True)
    parser.add_argument('--test-dataset-path', type=str, help='Testing dataset to use', required=True)
    parser.add_argument('--import-statement', type=str, help='Import statement to use to import model', required=True)
    parser.add_argument('--class-name', type=str, help='Class name of the model to instantiate', required=True)
    parser.add_argument('--parameters-to-test', type=json.loads, help='Arguments to test in hyperparameter search', required=True)
    parser.add_argument('--predict-classification-features', default=False, action='store_true', help='Predict the classification features instead of the regression ones')
    parser.add_argument('--fbeta-value', type=float, default=1, help='Beta for Fbeta macro in classification.')
    parser.add_argument('--max-rules', type=int, default=40, help='Maximum number of rules permitted by rule system', required=False)
    args = parser.parse_args()
    main(
        output_path=args.output_path,
        train_dataset_path=args.train_dataset_path,
        test_dataset_path=args.test_dataset_path,
        import_statement=args.import_statement,
        class_name=args.class_name,
        parameters_to_test=args.parameters_to_test,
        predict_classification_features=args.predict_classification_features,
        fbeta_value=args.fbeta_value,
        max_rules=args.max_rules,
    )
