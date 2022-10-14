# Categorize PEs by Imports

## Intro
This project attempts to categorize Portable Executable files
(the binary format used by Microsoft Windows executables) as
malware / benign-ware based on the number of functions imported
from common dynamically linked libraries. This is classification
of labeled data.

New malware approaches appear every day and constant updates to
software mean that new malicious executables can appear at any
time. Some part of the behavior of an executable is defined by
the libraries that it uses. Weak indicators of malicious behavior
can be combined to identify executables that otherwise might not
be detected.

When learning to reverse-engineer malware (many years ago), I was
told that there are imported functions and libraries that could
be indicators for further investigation. Given a dataset that
includes import information, I wanted to know how well different
models would perform on a small part of the information given,
specifically how many functions were imported from a small set of
common dynamically linked libraries.

### Dataset Overview
The "EMBER" dataset includes hundreds of features (per executable)
extracted from 1.1M Portable Executable files

- 900K training samples (evenly malware, benign-ware, and unlabelled)
- 200K test samples (evenly malware and benign-ware)

Data is Available at: https://github.com/elastic/ember

On that public github site, the EMBER project is described as:

"EMBER: a labeled benchmark dataset for training machine learning models to statically detect malicious Windows portable executable files."

H. Anderson and P. Roth, "EMBER: An Open Dataset for Training Static PE Malware Machine Learning Models”, in ArXiv e-prints. Apr. 2018.

### Structure of the Dataset
- `jsonl` files where each line is a JSON document describing a single PE file
- many features of each PE file are included (approaching 1k
  fields/features/columns for each)
- Features to chose from included:
  - Header fields describing the PE file's structure
  - Byte frequencies for the entire PE file
  - Entropy values for sections of the PE file
  - All libraries imported by the PE file, along with the functions from each

### Chosen Features (Import Table Values)
Import tables in portable executables define the functions
explicitly used from a dynamically loaded library (others can be
used and those may be better indicators of malicious intent...
but would require a different analysis approach, that is worth
combining with the one attempted here).

## Data Preprocessing
For simplicity, I decided to limit the analysis and feature
selection to only the import table fields. This required
parsing these fields out of the source data, a simple lengthy
process to extract from the `jsonl` files. This revealed 61180
unique file names appeared in the import sections of the
training set after deduplicating for capitalization.

### Common Import Frequency
- Decided that this could be limited to the most common files (to reduce
  feature count to a more manageable/explainable number)
- Limited to the 69 most common file names

The histogram below shows the long tail of import file names (after
using case-lowering to deduplicate):

![Common Import File Frequency](graphs/feature-count-hist-common.png "Common File Import Frequency")

Considered two different ways to define "common" imports:

- number of times the imported library was imported over the training set:
  imported_file_frequency.csv
- number of functions that were imported from the library over the training set:
  imported_file_counts.csv

Preprocessed the training data out of the original JSON lines into a
single Comma Separated Value (CSV) file (`training_import_features.csv`)
Containing 900,000 lines of training data, where each line contained
the number of functions imported from each of the commonly imported
libraries as comma separated entries.

### Down-selecting Features
Most of the chosen cleaning was about "down-selecting" to have
a smaller number of features. This was done to accommodate the
limited computational resources available and to my understanding
of the methods chosen.

## Data Exploration

### Limiting Subset of Features
For the purposes of statistical analysis (and visual comparison)
it is simpler to limit the features even further (beyond what is
used for the model building to follow). This is because of the
sparseness of the matrix (log-tail of common files).

### Pair Plot
![Pair Plot of Features](graphs/feature-pairplot.png)

### Correlation of Features
![Feature Correlation](graphs/feature-heatmap.png)

### Feature Importance
![Feature Importance](graphs/feature-importance.png)

## Building, Improving, and Choosing Models
Based on the models we looked at during this course, I chose the
following models to compare for classification

- LogisticRegression
- DecisionTreeClassifier
- RandomForestClassifier
- AdaBoostClassifier
- SVM (Dropped due to how long it ran)

### Comparison of techniques
- using initial (naive) hyperparameters
- train_test_split of training data

| Model                  | accuracy | recall | f1_score | cross_val mean | cross_val std |
| ---------------------- | -------- | ------ | -------- | -------------- | ------------- |
| LogisticRegression     | 0.646    | 0.587  | 0.624    | 0.58           | 0.071         |
| DecisionTreeClassifier | 0.911    | 0.913  | 0.911    | 0.62           | 0.12          |
| RandomForestClassifier | 0.826    | 0.876  | 0.834    | 0.72           | 0.12          |
| AdaBoostClassifier     | 0.889    | 0.873  | 0.887    | 0.59           | 0.099         |

### Improving Hyper-parameters
`GridSearchCV` was used to find better values for selected
hyperparameters for each of the classification techniques that
were compared.

The "best parameters" selected by the grid are then used to create new
models that are subjected to the same comparisons used with the naive
parameters.

#### LogisticRegression
'best parameters': {'C': 2.0, 'penalty': 'l1', 'solver': 'liblinear'}

#### DecisionTreeClassifier
'best parameters': {'criterion': 'entropy', 'max_features': 'sqrt'}

<!----
################################################
##### TODO ADD Parameters for RandomForest #####
################################################

#### RandomForestClassifier
'best parameters':
-->

<!----
######################################################
##### TODO ADD Parameters for AdaBoostClassifier #####
######################################################

#### AdaBoostClassifier
'best parameters':
-->

<!----
#############################################################
##### TODO ADD Evaluations of RandomForest and AdaBoost #####
#############################################################
-->

#### Effectiveness after improving the Hyper-parameters
| Model                  | accuracy | recall | f1_score | cross_val mean | cross_val std |
| ---------------------- | -------- | ------ | -------- | -------------- | ------------- |
| LogisticRegression     | 0.73     | 0.886  | 0.766    | 0.71           | 0.037         |
| DecisionTreeClassifier | 0.91     | 0.913  | 0.91     | 0.62           | 0.085         |
| RandomForestClassifier |          |        |          |                |               |
| AdaBoostClassifier     |          |        |          |                |               |

### Results when models are run on test data
While developing models, portions of the training data were used to
score the models with `train_test_split`, `cross_val_score`, or
`GridSearchCV` that let me validate using subsets of the training data.


<!----
##########################################
##### TODO ADD Final Evaluation HERE #####
##########################################
Now that I have some models that seem to work reasonably effectively,
it is time to train over the whole training set and then test using
the held-out test set.
-->