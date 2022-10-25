#!/bin/bash

#SBATCH -t 30:00
#SBATCH --constraint=rhel8
#SBATCH --ntasks-per-node=20
echo "Normal Job (Not in debug partition)"

echo "multiCNN More run100"
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=1"
export OMP_NUM_THREADS=1
bash timing.bash > times
python3 ../../../evaluate/avg.py times
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=2"
export OMP_NUM_THREADS=2
bash timing.bash > times
python3 ../../../evaluate/avg.py times
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=4"
export OMP_NUM_THREADS=4
bash timing.bash > times
python3 ../../../evaluate/avg.py times
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=8"
export OMP_NUM_THREADS=8
bash timing.bash > times
python3 ../../../evaluate/avg.py times
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=16"
export OMP_NUM_THREADS=16
bash timing.bash > times
python3 ../../../evaluate/avg.py times
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=32"
export OMP_NUM_THREADS=32
bash timing.bash > times
python3 ../../../evaluate/avg.py times
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=64"
export OMP_NUM_THREADS=64
bash timing.bash > times
python3 ../../../evaluate/avg.py times
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=128"
export OMP_NUM_THREADS=128
bash timing.bash > times
python3 ../../../evaluate/avg.py times
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"