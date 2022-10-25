#!/bin/bash

#SBATCH -t 02:00
#SBATCH --constraint=rhel8
#SBATCH --ntasks-per-node=20

echo "run100"
echo "/////////////////////////////////////////////////////////////"
echo "CUDA"
bash timing.bash > times
python3 ../../../evaluate/avg.py times
cat ../acc.csv
echo "/////////////////////////////////////////////////////////////"