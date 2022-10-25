bash timing.bash > times
python3 ../../../evaluate/avg.py times
cat ../acc.csv
