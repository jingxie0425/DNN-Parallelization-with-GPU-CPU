#!/bin/bash

#SBATCH -t 02:00
#SBATCH --constraint=rhel8
#SBATCH --ntasks-per-node=20

echo "run1"
echo "/////////////////////////////////////////////////////////////"
echo "CUDA"
./driver > save
python3 ../../../evaluate/addTime.py save
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"