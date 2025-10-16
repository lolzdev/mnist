#!/bin/bash

curl -L -o mnist-dataset.zip https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset
mkdir dataset
pushd dataset
unzip ../mnist-dataset.zip
rm -rf ../mnist-dataset.zip
popd
