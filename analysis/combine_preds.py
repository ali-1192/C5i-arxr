#!/Users/peter.vacca/.pyenv/shims/python

import glob
import os
import pandas as pd
from datetime import date
pd.set_option('max_colwidth', 800)

today = date.today().strftime("%Y-%m-%d")

def combine_preds(query):
    preds = glob.glob(f'../data/predictions/{query}/*')
    data = []
    for path in preds:
        df = pd.read_json(path,orient='records',lines=True)
        data.append(df)
    all_cat_preds = pd.concat(data).reset_index(drop=True)
    all_cat_preds['query'] = query
    all_cat_preds.to_json(f'../data/predictions/final_preds/{query}.json',orient='records',lines=True)


for query in ['headsets', 'categories', 'glasses']:
    combine_preds(query)

all_predictions = glob.glob('../data/predictions/final_preds/*')
data = []
for path in all_predictions:
    df = pd.read_json(path,orient='records',lines=True)
    data.append(df)
all_preds = pd.concat(data).reset_index(drop=True)
all_preds.to_json('../data/predictions/final_preds/all_preds.json',orient='records',lines=True)