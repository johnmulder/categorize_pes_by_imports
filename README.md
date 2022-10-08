# Categorize PEs by Imports

## Overview
This project attempts to categorize Portable Executable files
as malware / benign-ware based on the number of
functions imported from common dynamically linked libraries.

## Setup

### Download the ember source data
`git clone https://github.com/elastic/ember.git`

### Install prerequisites
`pip3 install -r requirements.txt`

## Run Tools

### Extracting and cleaning data from the ember dataset
- `python3 01_data_cleaning.py -d ember/ember_data -o training`
- `mkdir artifacts`
- `vm training* artifacts/`


