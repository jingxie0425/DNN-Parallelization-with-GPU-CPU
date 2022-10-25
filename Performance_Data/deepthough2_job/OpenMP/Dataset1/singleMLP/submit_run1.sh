#!/bin/bash

#SBATCH -t 10:00
#SBATCH --constraint=rhel8
#SBATCH --ntasks-per-node=20
echo "Normal Job (Not in debug partition)"

echo "singleMLP run1"
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=1"
export OMP_NUM_THREADS=1
./driver > save
python3 ../../../evaluate/addTime.py save
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=2"
export OMP_NUM_THREADS=2
./driver > save
python3 ../../../evaluate/addTime.py save
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=4"
export OMP_NUM_THREADS=4
./driver > save
python3 ../../../evaluate/addTime.py save
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=8"
export OMP_NUM_THREADS=8
./driver > save
python3 ../../../evaluate/addTime.py save
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=16"
export OMP_NUM_THREADS=16
./driver > save
python3 ../../../evaluate/addTime.py save
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=32"
export OMP_NUM_THREADS=32
./driver > save
python3 ../../../evaluate/addTime.py save
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=64"
export OMP_NUM_THREADS=64
./driver > save
python3 ../../../evaluate/addTime.py save
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"
echo "OMP_NUM_THREADS=128"
export OMP_NUM_THREADS=128
./driver > save
python3 ../../../evaluate/addTime.py save
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"