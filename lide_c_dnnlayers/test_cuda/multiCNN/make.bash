#!/bin/bash

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}$
export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

pushd ../../../cnnnsight
make clean;make all
popd
pushd ../../libs
make clean;make all
popd
pushd ../../actor
make clean;make all
popd
pushd ../../autoGenGraph_cnn
make clean;make all
popd
make clean;make all
./driver > save
pushd ../../../evaluate
python3 evaluate.py ../lide_c_dnnlayers/test_cuda/result.csv ../../graphGen/extractedParas/testInputY.csv ..//lide_c_dnnlayers/test_cuda/acc.csv
popd
python3 ../../../evaluate/addTime.py save
cat ../acc.csv

