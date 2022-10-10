"""
Exploring and characterizing the extracted features
from the ember dataset's training features by producing
graphs for the user to view and evaluate.

Expects there to be a an input folder containing:
  - Feature frequency file ending in "*-frequency.csv"
  - Feature training file ending in "*-values.parquet"

Exported files include:
    - "graphs/feature-heatmap.png"
    - "graphs/feature-importance.png"
    - "graphs/feature-pairplot.png"
    - "graphs/feature-numeric-<feature_name>.png"

Example Use:
    python3 02_data_exploration.py -i artifacts -o graphs
"""
import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

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
        help="input directory path",
        default="graphs",
    )
    args = parser.parse_args()
    feature_freq_file = next(args.input.glob("*-frequency.csv"))
    training_file = next(args.input.glob("*-values.parquet"))

    top_common_libs = utils.get_common_feature_names(feature_freq_file, n=10)
    feature_df = utils.get_training_df(training_file)
    common_feature_df = utils.get_selected_features(
        feature_df, top_common_libs + ["label"]
    )

    print(f"## Feature Statistics")
    print(df_stats(common_feature_df))

    print("## Writing out histograms")
    plot_import_hist(feature_freq_file, args.output)
    plot_import_hist(feature_freq_file, args.output, common=top_common_libs)

    print("## Writing out pairplot of features")
    sns.pairplot(common_feature_df)
    plt.savefig(args.output / "feature-pairplot.png")

    print("## Writing out feature correlation heatmap of features")
    plt.figure(figsize=(18, 10))
    sns.heatmap(common_feature_df.corr(), annot=True)
    plt.savefig(args.output / "feature-heatmap.png")

    print("## Writing out feature plots for each features")
    plot_numeric(common_feature_df, top_common_libs, "label", args.output)
    plot_feature_importance(common_feature_df, args.output)


def df_stats(df):
    "returns a pandas Dataframe of key statistical properties that describe the columns of a Dataset"
    mean = pd.DataFrame(df.mean())
    unique = df.apply(lambda x: x.unique().shape[0])
    skew = df.skew()
    kurt = df.kurt()
    col_names = ["mean", "unique", "skewn", "kurt"]
    temp = pd.concat([mean, unique, skew, kurt], axis=1)
    temp.columns = col_names
    return temp


def plot_import_hist(freqs_fn, directory, common=None):
    "Writes histogram of import file frequency to a PNG"
    df = pd.read_csv(freqs_fn, names=["name", "count"], index_col="name")
    df.sort_values("count", inplace=True, ascending=False)
    if common is None:
        df.plot.bar()
        plt.savefig(directory / "feature-count-hist-all.png")
    else:
        df = df.loc[common]
        df.plot.bar()
        plt.tight_layout()
        plt.savefig(directory / "feature-count-hist-common.png")


def plot_numeric(df, cols, target, directory):
    "Writes distribution plots of an import file's frequency to a PNG"
    for col in cols:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        sns.distplot(a=df[col], ax=ax[0])
        ax[0].set_title(f"distribution of {col}, skew={df[col].skew():.4f}")
        sns.boxenplot(data=df, x=target, y=col, ax=ax[1])
        ax[1].set_title("Boxen Plot Split by Target")
        plt.savefig(directory / f"feature-numeric-{col}.png")


def plot_feature_importance(df, directory):
    clf = RandomForestClassifier()
    features = df.drop("label", axis=1).values
    clf.fit(features, df["label"].values)
    fig = plt.figure(figsize=(18, 8))
    importance = pd.Series(
        clf.feature_importances_, index=df.drop("label", axis=1).columns
    ).sort_values(ascending=False)
    sns.barplot(x=importance, y=importance.index)
    plt.title("Feature Importance")
    plt.xlabel("Score")
    plt.savefig(directory / "feature-importance.png")


if __name__ == "__main__":
    main()
