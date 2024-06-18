#!/usr/bin/env python3
import os

queries = ['headsets', 'glasses', 'categories']
dirs = ['data', 'predictions']

for dir in dirs:
    os.makedirs(f'./{dir}', exist_ok=True)

for query in queries:
    os.makedirs(f'./data/{query}', exist_ok=True)
    os.makedirs(f'./predictions/{query}', exist_ok=True)

os.makedirs('./predictions/final_preds', exist_ok=True)
