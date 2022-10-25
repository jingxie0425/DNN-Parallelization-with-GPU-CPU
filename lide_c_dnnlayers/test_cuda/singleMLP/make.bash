pushd ../../../cnnnsight
make clean;make all
popd
pushd ../../actor
make clean;make all
popd
pushd ../../libs
make clean;make all
popd
pushd ../../graph
make clean;make all
popd
make clean;make all
./driver
pushd ../../../evaluate
python3 evaluate.py ../lide_c_dnnlayers/test_cuda/result.csv ../extractParas_mod4/testInputY.csv ..//lide_c_dnnlayers/test_cuda/acc.csv
popd
