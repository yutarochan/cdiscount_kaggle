'''
Dataset Utility CLI & Library
Contains various auxillary functions for dataset management and loading.

Source: https://www.kaggle.com/humananalog/keras-generator-for-reading-directly-from-bson
Author: Yuya Jeremy Ong (yjo5006@psu.edu)
'''
from __future__ import print_function
import io
import os
import sys
import math
import bson
import struct
from tqdm import *
import numpy as np
import pandas as pd
import multiprocessing as mp

# Application Parameters
DATA_DIR = '/backups4/emotion/.data/'

TRAIN_BSON_PATH = os.path.join(DATA_DIR, "train.bson")
TEST_BSON_PATH  = os.path.join(DATA_DIR, "test.bson")

NUM_TRAIN_PROD = 7069896
NUM_TEST_PROD  = 1768182

# Generate Category Lookup CSV Table
def gen_lookupcsv(input_file, output_file):
    cat_path = os.path.join(DATA_DIR, input_file)
    cat_df = pd.read_csv(cat_path, index_col="category_id")
    cat_df["category_idx"] = pd.Series(range(len(cat_df)), index=cat_df.index)
    cat_df.to_csv(DATA_DIR + output_file)

# Build Lookup Dictionary for Categories
def build_catlookup(input_file):
    cat2idx = {}
    idx2cat = {}
    cat_df = pd.read_csv(input_file, index_col="category_id")
    for ir in cat_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

# BSON Data Loader
# TODO: Configure this to perform the computation in parallel.
def read_bson(bson_path, num_records, with_categories):
    rows = {}
    with open(bson_path, "rb") as f, tqdm(total=num_records) as pbar:
        offset = 0
        while True:
            item_length_bytes = f.read(4)
            if len(item_length_bytes) == 0: break

            length = struct.unpack("<i", item_length_bytes)[0]

            f.seek(offset)
            item_data = f.read(length)
            assert len(item_data) == length

            item = bson.BSON(item_data).decode()
            product_id = item["_id"]
            num_imgs = len(item["imgs"])

            row = [num_imgs, offset, length]
            if with_categories: row += [item["category_id"]]
            rows[product_id] = row

            offset += length
            f.seek(offset)
            pbar.update()

    columns = ["num_imgs", "offset", "length"]
    if with_categories: columns += ["category_id"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "product_id"
    df.columns = columns
    df.sort_index(inplace=True)
    return df

# Generate Item Offset and Length Metadata CSV
def gen_trainoffsetcsv(report_stat=False):
    train_offsets_df = read_bson(TRAIN_BSON_PATH, num_records=NUM_TRAIN_PROD, with_categories=True)

    if report_stat:
        print('PRODUCT COUNT: ' + str(len(train_offsets_df)))
        print('PRODUCT CATEGORY COUNT: ' + str(len(train_offsets_df["category_id"].unique())))
        print('TOTAL IMAGE COUNT: ' + str(train_offsets_df["num_imgs"].sum()))

    train_offsets_df.to_csv(DATA_DIR + "train_offsets.csv")

if __name__ == '__main__':
    # gen_lookupcsv('category_names.csv', 'categories.csv')   # Generate Lookup CSV File
    gen_trainoffsetcsv(True)    # Generate Item Offset CSV File
