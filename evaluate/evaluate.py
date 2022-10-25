'''
@author xiaomin wu
'''
import csv 
import sys

resultfilePath = sys.argv[1]
labelfilePath = sys.argv[2]
outPath = sys.argv[3]

result = []
groundTruth = []

correctCount = 0
totalCount = 0

with open(resultfilePath) as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader:
		result.append(int(row[0]))
csvfile.close()

with open(labelfilePath) as csvfile:
	spamreader = csv.reader(csvfile, delimiter=',')
	for row in spamreader:
		groundTruth.append(int(float(row[0])))
csvfile.close()

totalCount = len(groundTruth)
for i in range(len(groundTruth)):
    if result[i] == groundTruth[i]:
        correctCount = correctCount + 1
    
acc = (float(correctCount)/totalCount) * 100

with open(outPath, 'w+', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)   
    spamwriter.writerow(["Accuracy: "+str(acc)])