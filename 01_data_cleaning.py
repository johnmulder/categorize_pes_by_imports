"""
Extracting and cleaning data from the ember dataset
to build machine learning models using the number
of imported functions from commonly imported libraries

Exported files (with default "out" stem)include:
    - out-values.parquet     The feature values extracted as a parquet
    - out-names.txt      Names of all of the features (one per line)
    - out-counts.csv     Features with the summed values for the data set
    - out-frequency.csv  Features with the number of times each occurred

Example Uses:
    python3 01_data_cleaning.py -f ember/ember_data/train_features_0.jsonl
    python3 01_data_cleaning.py -d ember/ember_data -o training
"""
import argparse
import json
import pathlib
from collections import Counter
from multiprocessing import Process, Queue

import pandas as pd


def main():
    "driver function for script"
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", type=pathlib.Path)
    parser.add_argument("-f", "--file", type=pathlib.Path)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="output file stem (without extension)",
        default="out",
    )
    args = parser.parse_args()
    if args.directory:
        in_files = get_files(args.directory, prefix="train")
    elif args.file:
        in_files = [args.file]
    else:
        print("must give either -d or -f")
        parser.print_help()
        parser.exit()
    feature_list, names, counts, frequency = extract_features(in_files)
    write_dict_list(feature_list, f"{args.output}-values.parquet")
    write_feature_list(names, f"{args.output}-names.txt")
    write_dict(counts, f"{args.output}-counts.csv")
    write_dict(frequency, f"{args.output}-frequency.csv")


def extract_features(in_files):
    """
    Takes data input files and extracts the features
    of interest.
    Operates by firing off a new process for each file.
    Returns:
        total_column_names
        total_column_counts
        total_column_frequency
    """
    # Initialize the result aggregation containers
    total_feature_dict_list = []
    total_column_names = set()
    total_column_counts = Counter()
    total_column_frequency = Counter()
    print("Firing off processes (one per file) to create feature files")
    processes = []
    results_queue = Queue()
    for file_path in in_files:
        process = Process(target=feat_summary_task, args=(file_path, results_queue))
        processes.append(process)
        process.start()
    print("Aggregating process results")
    for process in processes:
        (
            feature_dict_list,
            column_names,
            column_counts,
            column_feqs,
        ) = results_queue.get()  # get has to happen before join
        total_feature_dict_list.extend(feature_dict_list)
        total_column_names.update(column_names)
        total_column_counts.update(column_counts)
        total_column_frequency.update(column_feqs)
        print("Totals from a process have been combined")
    for process in processes:
        process.join()  # this blocks until the process terminates
        print("A process has finished/joined")
    # Report out the aggregate results
    print(f"got features for {len(total_feature_dict_list)} different samples")
    # Remove the uncommon features from the total_feature_dict_list
    common_columns = {c[0] for c in total_column_counts.most_common(100)}
    common_columns.add("label")
    reduced_feat_dicts = [reduce(d, common_columns) for d in total_feature_dict_list]
    return (
        reduced_feat_dicts,
        total_column_names,
        total_column_counts,
        total_column_frequency,
    )


def get_files(directory, prefix="train"):
    """
    Returns files from directory, optionally
    selects only those with a specific prefix
    """
    dir_path_iter = pathlib.Path(directory).iterdir()
    if prefix:
        files = (p for p in dir_path_iter if str(p.stem).startswith(prefix))
    else:
        files = (p for p in dir_path_iter)
    return files


def feat_summary_task(file, result_queue):
    """
    Runs `get_feat_summary` over the files and
    returns results into a `result_queue`
    """
    feature_dict_list, column_names, column_counts, column_feqs = get_feat_summary(file)
    result_queue.put((feature_dict_list, column_names, column_counts, column_feqs))


def get_feat_summary(file_path):
    """
    Pulls out features from a newline-delimited JSON file.
    Returns:
        feature_dict_list
        column_names
        column_counts
        column_feqs
    """
    feature_dict_list = []
    column_names = set()
    column_counts = Counter()
    column_feqs = Counter()
    counter = 0
    with open(file_path, "rt", encoding="utf-8") as in_file:
        for line in in_file.readlines():
            counter += 1
            feature_dict = get_features(json.loads(line))
            feature_dict_list.append(feature_dict)
            column_counts.update(feature_dict)
            column_feqs.update({k: 1 for k in feature_dict})
            column_names = column_names.union(feature_dict.keys())
            if counter % 10000 == 0:
                print(f"{file_path.name} : processed {counter}")
    return feature_dict_list, column_names, column_counts, column_feqs


def get_features(pe_info):
    "Extracts import data and label from a pe entry"
    features = {}
    features.update(get_import_data(pe_info))
    features.update({"label": pe_info["label"]})
    return features


def get_import_data(pe_info):
    "Extracts import data from a PE entry"
    return {file.lower(): len(funcs) for file, funcs in pe_info["imports"].items()}


def reduce(orig_dict, keys_to_keep):
    "remove all keys not in keys_to_keep from orig_dict"
    return {k: v for k, v in orig_dict.items() if k in keys_to_keep}


def write_feature_list(column_names, out_file_name):
    "Writes out a list of features to a text file, one name per line."
    print(f"got {len(column_names)} imported files combined across those")
    with open(out_file_name, "wt", encoding="utf-8") as out_file:
        for column_name in column_names:
            out_file.write(column_name + "\n")


def write_dict(feature_dict, out_file_name):
    "Writes out a dictionary into a CSV file"
    with open(out_file_name, "wt", encoding="utf-8") as csv_file:
        for name in feature_dict:
            if name not in ("id", "label"):
                # Wrap the name in quotes to defend against commas in them
                csv_file.write(f'"{name}", {feature_dict[name]}\n')


def write_dict_list(dict_list, out_file_name):
    "Writes out a list of dictionaries into a parquet file"
    print("Attempting to create DataFrame from dictionaries")
    df = pd.DataFrame(dict_list)
    print("Successfuly created DataFrame from dictionaries")
    df.fillna(0, inplace=True)
    print("Starting to wite out as parquet file")
    df.to_parquet(out_file_name)


if __name__ == "__main__":
    main()
