import csv
import sys

resultfilePath = sys.argv[1]
result = 0
count = 0
with open(resultfilePath) as csvfile:
	spamreader = csv.reader(csvfile, delimiter=' ')
	for row in spamreader:
		count = count+1
		result = result + float(row[3])
csvfile.close()
result = (float)(result/count)
print("Avg calculation runtime: "+str(result)+" ms")