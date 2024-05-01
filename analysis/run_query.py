#!/Users/pvacca/.pyenv/shims/python

import glob
from utils import make_preds
import pandas as pd

query='headsets'
tags = ['pico',
 'samsung',
 'sony',
 'valve']
querys = [query for i in range(len(tags))]

# for query,tag in zip(querys,tags):
#     print(f"{query} - {tag}")
#     make_preds(query,tag)


_predictions = glob.glob(f'../data/predictions/{query}/*')
data = []
for path in _predictions:
    df = pd.read_json(path,orient='records',lines=True)
    data.append(df)
all_preds = pd.concat(data).reset_index(drop=True)
all_preds['query'] = query
all_preds.to_json(f'../data/predictions/final_preds/{query}.json',orient='records',lines=True)