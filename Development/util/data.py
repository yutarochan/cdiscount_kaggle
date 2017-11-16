'''
Dataset Utility CLI & Library
Contains various auxillary functions for dataset management and loading.

Sources Utilized:
- https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson
- https://www.kaggle.com/theblackcat/loading-bson-data-for-keras-fit-generator/notebook

Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import io
import os
import sys
import math
import bson
import timeit
import struct
from tqdm import *
import numpy as np
import pandas as pd
import multiprocessing as mp

# Application Parameters
DATA_DIR = '/storage/work/yjo5006/data/'

TRAIN_SAMPLE_PATH = os.path.join(DATA_DIR, "train_example.bson")
TRAIN_BSON_PATH = os.path.join(DATA_DIR, "train.bson")
TEST_BSON_PATH  = os.path.join(DATA_DIR, "test.bson")
CATEGORY_PATH = os.path.join(DATA_DIR, "categories.csv")

NUM_TRAIN_PROD = 7069896
NUM_TEST_PROD  = 1768182

CAT_SIZE = 5270

# Generate Category Lookup CSV Table
def build_lookup(input_file, output_file):
    cat_path = os.path.join(DATA_DIR, input_file)
    cat_df = pd.read_csv(cat_path, index_col="category_id")
    cat_df["category_idx"] = pd.Series(range(len(cat_df)), index=cat_df.index)
    cat_df.to_csv(DATA_DIR + output_file)

# Build Lookup Dictionary for Categories
def build_cat(input_file):
    cat2idx = {}
    idx2cat = {}
    cat_df = pd.read_csv(input_file, index_col="category_id")
    for ir in cat_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

# Category to One-Hot Encoded Vector
def hot_cat(cat, cat2idx):
    return True, ([0] * CAT_SIZE)[cat2idx[cat]]

# Keras Data Generator
def data_generator(batch_size=128, st_idx=0):
    data = bson.decode_file_iter(open(TRAIN_BSON_PATH, 'rb'))
    cnt_prod = 0
    X = []
    y = []
    while True:
        cnt = 0
        for d in data:
            # Skip to Start Index
            if cnt_prod < st_idx:
                cnt_prod += 1
                continue

            # One Hot Encoding Index Conversion from Category Data
            success, one_hot = hot_cat(d['category_id'])

            # Pass if ID Conversion Failed
            if not success:
                print('ID Conversion Failed')
                continue

            # Iterate through Each Picture Sample
            for pic in d['imgs']:
                X.append(imread(io.BytesIO(pic['picture'])))
                y.append(one_hot)
                cnt += 1

            if cnt >= batch_size:
                cnt = 0
                X = np.asarray(X)
                y = np.asarray(y)

                # Perform Shuffling Operation (Every Day I'm Shufflin')
                for i, im in enumerate(X[:int(batch_size/2)]):
                    j = randint(0, batch_size-1)
                    y_temp = y[i]
                    img_temp = im
                    X[i] = X[j]
                    y[i] = y[j]
                    X[j] = img_temp
                    y[j] = y_temp

                # Yield Final Batch Result
                yield X, y

                # Clear Off Batch Memory
                del X
                del y
                X = []
                y = []


if __name__ == '__main__':
    # gen_lookupcsv('category_names.csv', 'categories.csv')   # Generate Lookup CSV File
    cat = build_cat(CATEGORY_PATH)
    gen = data_generator()
    next(train_gen)

    start = timeit.timeit()
    bx, by = next(train_gen)
    end = timeit.timeit()

    print(end - start)
