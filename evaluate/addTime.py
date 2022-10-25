import csv
import sys

resultfilePath = sys.argv[1]
result = 0;
with open(resultfilePath) as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ')
	for row in spamreader:
		result = result + float(row[2])
csvfile.close()
print("Total calculation runtime: "+str(result)+" ms")