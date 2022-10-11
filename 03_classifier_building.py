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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

import utils


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
        help="output file path",
        default="classification_evaluation.md",
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
    with open(args.output, "wt", encoding="utf-8") as out_file:
        print_metrics_header(out_file)
        assess_naive_classifiers(X, y, out_file)


def assess_naive_classifiers(X, y, out_file):
    """
    Trying the data on a few different classifiers
    (with naively chosen hyper-parameters)
    """
    assess_classifier(X, y, LogisticRegression(), out_file)
    assess_classifier(X, y, DecisionTreeClassifier(), out_file)
    assess_classifier(X, y, RandomForestClassifier(), out_file)
    assess_classifier(X, y, AdaBoostClassifier(), out_file)


def assess_classifier(X, y, clf, out_file):
    print(f"## Evaluating Naive {type(clf).__name__}")
    # create test-train split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # fit and predict
    if isinstance(clf, LogisticRegression):
        warnings.filterwarnings("ignore")
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    # evaluate the effectiveness of the classifier
    print_metrics(y_test, predictions, clf, out_file)
    # print(f"cross_val_score: {cross_val_score(clf, X, y)}")
    scores = cross_val_score(clf, X, y)
    out_file.write(f"| {scores.mean():.2} | {scores.std():.2}")
    # output plot of Confusion Matrix
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    display.plot()


def evaluate_params_for_classifers(X, y, out_file):
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
            "max_depth": [2**x for x in (2, 3)] + [None],
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
    for classifier_class, params in params_by_class.items():
        # suppressing all warnings, because GridSearchCV throws a lot
        # of unhelpful warnings
        warnings.filterwarnings("ignore")
        best_params = evaluate_params(X, y, classifier_class, params)
        clf = classifier_class(**best_params)
        print("### Metrics with the 'best parameters'")
        assess_classifier(X, y, clf, out_file)


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


def print_metrics_header(out_file):
    out_file.write(
        "| Classifier | Accuracy | Recall | F1 Score | cross_val_score | cross_val_std |"
    )


def print_metrics(y_true, y_pred, clf, out_file):
    out_file.write(f"| {type(clf).__name__} ")
    out_file.write(f"| {accuracy_score(y_true, y_pred):.3} ")
    out_file.write(f"| {recall_score(y_true, y_pred):.3} ")
    out_file.write(f"| {f1_score(y_true, y_pred):.3} ")


if __name__ == "__main__":
    main()
