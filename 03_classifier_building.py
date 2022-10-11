"""
Building an evaluating classifiers
from the ember dataset's training features by producing
graphs for the user to view and evaluate.

Expects there to be a an input folder containing:
    -

Exported files include:
    -

Example Use:
    python3 03_classifier_building.py
"""
import argparse
import pathlib
import warnings
from pprint import pprint

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, recall_score)
from sklearn.model_selection import (GridSearchCV, cross_val_score,
                                     train_test_split)
from sklearn.tree import DecisionTreeClassifier

import utils

# TODO: suppress ConvergenceWarning when training LogisticRegression
# TODO: Refactor evaluation output to be a table
# TODO: Refactor output to a markdown file

def main():
    "driver function for script"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=pathlib.Path,
        help="input directory path",
        default="artifacts",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        help="input directory path",
        default="artifacts",
    )
    args = parser.parse_args()
    feature_freq_file = next(args.input.glob("*-frequency.csv"))
    training_file = next(args.input.glob("*-values.parquet"))

    top_common_libs = utils.get_common_feature_names(feature_freq_file, n=10)
    feature_df = utils.get_training_df(training_file)
    common_feature_df = utils.get_selected_features(
        feature_df, top_common_libs + ["label"]
    )
    X, y = utils.recover_features(common_feature_df)
    assess_naive_classifiers(X, y)


def assess_naive_classifiers(X, y):
    """
    Trying the data on a few different classifiers
    (with naively chosen hyper-parameters)
    """
    assess_classifier(X, y, LogisticRegression())
    assess_classifier(X, y, DecisionTreeClassifier())
    assess_classifier(X, y, RandomForestClassifier())
    assess_classifier(X, y, AdaBoostClassifier())


def assess_classifier(X, y, clf):
    print(f"## Evaluating Naive {type(clf).__name__}")
    # create test-train split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # fit and predict
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    # evaluate the effectiveness of the classifier
    print_metrics(y_true=y_test, y_pred=predictions)
    # print(f"cross_val_score: {cross_val_score(clf, X, y)}")
    scores = cross_val_score(clf, X, y)
    print(f"- cross_val_score {scores.mean():.2} std: {scores.std():.2}")

    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    display.plot()


def evaluate_params_for_classifers(X, y):
    params_by_class = {
        LogisticRegression: {
            "penalty": ["l1", "l2", "elasticnet", "none"],
            "C": np.arange(0.5, 3.0, 0.5),
            "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        },
        DecisionTreeClassifier: {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_features": ["sqrt", "log2", None],
        },
        RandomForestClassifier: {
            "n_estimators": [10**x for x in (2, 3)],
            "max_depth": [
                2**x
                for x in (
                    2,
                    3,
                )
            ]
            + [None],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_features": ["sqrt", "log2", None],
        },
        AdaBoostClassifier: {
            "base_estimator": (),
            "max_depth": [2**x for x in (2, 3, 4)] + [None],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_features": ["sqrt", "log2", None],
        },
    }
    for classifier_class, params in params_by_class:
        # suppressing all warnings, because GridSearchCV throws a lot
        # of unhelpful warnings
        warnings.filterwarnings("ignore")
        best_params = evaluate_params(X, y, classifier_class, params)
        clf = classifier_class(**best_params)
        print("### Metrics with the 'best parameters'")
        assess_classifier(X, y, clf)


def evaluate_params(X, y, classifier_class, parameters):
    print(f"## Evaluating {classifier_class.name}")
    print("### Starting GridSearchCV")
    clf = classifier_class()
    grid = GridSearchCV(clf, parameters)
    print("### Created GridSearchCV")
    grid.fit(X, y)
    print("### Fit GridSearchCV")
    best_params = {}
    for param_name in parameters:
        best_params[param_name] = grid.best_params_[param_name]
    print("### Best Params Found")
    pprint(best_params)
    return best_params


def print_metrics(y_true, y_pred):
    print(f"- accuracy: {accuracy_score(y_true, y_pred):.3}")
    print(f"- recall: {recall_score(y_true, y_pred):.3}")
    print(f"- f1_score: {f1_score(y_true, y_pred):.3}")


if __name__ == "__main__":
    main()
