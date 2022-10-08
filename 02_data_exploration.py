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
# TODO:
#       - Build graphs of feature characterization
#         (including those not in this file)
import argparse
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier


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

    top_common_libs = get_common_feature_names(feature_freq_file, n=10)
    feature_df = get_training_df(training_file)
    common_feature_df = get_selected_features(feature_df, top_common_libs + ["label"])
    print(df_stats(common_feature_df))

    #plot_import_hist(feature_freq_file, args.output)
    plot_import_hist(feature_freq_file, args.output, common=top_common_libs)

    # sns.pairplot(common_feature_df)
    # plt.savefig(args.output / 'feature-pairplot.png')
    # plt.figure(figsize=(18, 10))
    # sns.heatmap(common_feature_df.corr(), annot=True)
    # plt.savefig(args.output / 'feature-heatmap.png')
    # plot_numeric(common_feature_df, top_common_libs, "label", args.output)
    # plot_feature_importance(common_feature_df, args.output)
    pass

def df_stats(df):
    mean = pd.DataFrame(df.mean())
    unique = df.apply(lambda x: x.unique().shape[0])
    skew = df.skew()
    kurt = df.kurt()

    col_names = ["mean", "unique", "skewn", "kurt"]
    temp = pd.concat([mean, unique, skew, kurt], axis=1)
    temp.columns = col_names
    return temp


def get_common_feature_names(freqs_fn, n=10):
    "get the n most common names in a frequency file as a list"
    lib_freq_df = pd.read_csv(freqs_fn, names=["name", "count"], index_col="name")
    sorted_df = lib_freq_df.sort_values("count", ascending=False)
    return sorted_df.head(n).index.tolist()


def get_training_df(feature_file):
    "get the training data from feature_file as a DataFrame (excluding rows with the label -1)"
    feature_df = pd.read_parquet(feature_file)
    return feature_df[feature_df.label != -1]


def get_selected_features(df, features):
    "reduce the feature Dataframe to only use the chosen columns"
    return df.loc[:, features]


def plot_import_hist(freqs_fn, directory, common=None):
    df = pd.read_csv(freqs_fn, names=["name", "count"], index_col="name")
    df.sort_values("count", inplace=True, ascending=False)
    if common is None:
        df.plot.bar()
        plt.savefig(directory / f'feature-count-hist-all.png')
    else:
        df = df.loc[common]
        df.plot.bar()
        plt.tight_layout()
        plt.savefig(directory / f'feature-count-hist-common.png')


def plot_numeric(df, cols, target, directory):
    for col in cols:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        sns.distplot(a=df[col], ax=ax[0])
        ax[0].set_title("distribution of {}, skew={:.4f}".format(col, df[col].skew()))
        sns.boxenplot(data=df, x=target, y=col, ax=ax[1])
        ax[1].set_title("Boxen Plot Split by Target")
        plt.savefig(directory / f'feature-numeric-{col}.png')


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
    plt.savefig(directory / 'feature-importance.png')


if __name__ == "__main__":
    main()
