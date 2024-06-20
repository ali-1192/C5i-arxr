#!/usr/bin/env python3
import os

queries = ['headsets', 'glasses', 'categories']

os.makedirs('./data', exist_ok=True)
os.makedirs('./data/predictions', exist_ok=True)
os.makedirs('./data/predictions/final_preds', exist_ok=True)

for query in queries:
    os.makedirs(f'./data/{query}', exist_ok=True)
    os.makedirs(f'./data/predictions/{query}', exist_ok=True)