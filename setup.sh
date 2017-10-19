#!/bin/bash
# Dataset File Configuration Script
# Author: Yuya Jeremy Ong (yjo5006@psu.edu)

echo "Setup Dataset Files"
mkdir -p Data/raw/
cd Data/raw/

echo "Downloading Files... This might take some time, so grab yourself some coffee. :)"
wget -x --load-cookies ../../cookies.txt -P . -nH --cut-dirs=5 https://www.kaggle.com/c/cdiscount-image-classification-challenge/download/test.bson
wget -x --load-cookies ../../cookies.txt -P . -nH --cut-dirs=5 https://www.kaggle.com/c/cdiscount-image-classification-challenge/download/train.bson
wget -x --load-cookies ../../cookies.txt -P . -nH --cut-dirs=5 https://www.kaggle.com/c/cdiscount-image-classification-challenge/download/train_example.bson
wget -x --load-cookies ../../cookies.txt -P . -nH --cut-dirs=5 https://www.kaggle.com/c/cdiscount-image-classification-challenge/download/category_names.7z
wget -x --load-cookies ../../cookies.txt -P . -nH --cut-dirs=5 https://www.kaggle.com/c/cdiscount-image-classification-challenge/download/sample_submissions.7z

cd ../../
echo "Done"
