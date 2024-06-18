# project-google-arxr-analytics

# Overview 

#### Author: Peter Vacca
#### Date: 2024-06-17
#### Version: 0.1

    This will explain the steps required to take the manual export spreadsheets and process them locally.
    It should be used as a loose guide when creating a new pipeline within Vertex AI.

## Local Environment Setup

Note: This can be done in a virtualenv, docker container or on localhost.

1. pip3 install --upgrade pip
2. pip3 install -r requirements.txt
3. Ensure gcloud is installed and authenticated
4. in git dir, run ./mkdirs.py, to create proper directory structure

## Gathering the data

1. Download manual exports as csv to the data/$query
2. File should be named $Tag.csv (eg Amazon.csv)
3. Run ./process_csv $query$ for each
    * or ./run_all.sh

## Processing the data

1. 