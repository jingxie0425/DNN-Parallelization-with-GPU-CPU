pushd ../../../nn_c
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
bash runC.bash
popd
