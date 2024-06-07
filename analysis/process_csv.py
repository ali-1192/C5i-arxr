#!/Users/peter.vacca/.pyenv/shims/python

import os
from utils import make_preds, make_preds_categories
import pandas as pd
import sys
import concurrent.futures
pd.set_option('max_colwidth', 800)

# Get the query argument from the command line
query = sys.argv[1]

# Define the possible values for the query argument
possible_values = ['headsets', 'glasses', 'categories']

# Check if the query argument is valid
if query not in possible_values:
    print("Invalid query argument. Possible values are 'headsets', 'glasses', or 'categories'.")
    sys.exit(1)

def get_tags(query):
    # Get the value of the 'query' argument and save it to a variable
    directory = f"../data/{query}/"

    # Get all CSV files in the directory
    tags = [file[:-4] for file in os.listdir(directory) if file.endswith('.csv')]
    tags.sort()
    return tags

tags = get_tags(query)
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []
    method = make_preds
    for tag in tags:
        if query == 'categories':
            method = make_preds_categories
        futures.append(executor.submit(method, query, tag))
    # Wait for all the tasks to complete
    concurrent.futures.wait(futures)