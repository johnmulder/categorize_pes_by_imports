"""
Utility functions for multiple steps in the data analysis and machine learning
with the number of imported functions from commonly imported libraries
"""

import pandas as pd


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


def recover_features(df):
    "split the dataframe into standard y and X for machine learning applications"
    # Drop any row where the label is "-1"
    # "uncertain" label is not great for training
    df = df[df.label != -1]
    # Create a dataset using the standard names y and X
    y = df.pop("label").values
    X = df
    return X, y
