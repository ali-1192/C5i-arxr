#!/Users/peter.vacca/.pyenv/shims/python

import glob
import os
from utils import make_preds
import pandas as pd
import argparse
pd.set_option('max_colwidth', 800)

# Create argument parser
parser = argparse.ArgumentParser(description='Parse a single argument')

# Add argument
parser.add_argument('query', type=str, help='Query string to be saved')

# Parse the arguments
args = parser.parse_args()

# Get the value of the 'query' argument and save it to a variable
query = args.query
directory = f"../data/{query}/"

# Get all CSV files in the directory
tags = [file[:-4] for file in os.listdir(directory) if file.endswith('.csv')]
tags.sort()

for tag in tags:
    print(f"{query} - {tag}")
    make_preds(query, tag)

_predictions = glob.glob(f'../data/predictions/{query}/*')
data = []
for path in _predictions:
    df = pd.read_json(path,orient='records',lines=True)
    data.append(df)
all_preds = pd.concat(data).reset_index(drop=True)
all_preds['query'] = query
all_preds.to_json(f'../data/predictions/final_preds/{query}.json',orient='records',lines=True)