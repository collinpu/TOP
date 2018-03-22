import tensorflow as tf
import re
import numpy as np
from testDatasetUtils import * 
from definitions import *
import sys

#bestParameters = [11,5,0.00001]
#bestArchitecture = [4,1024,0.77]

bestArchitecture = [5,1162,1.25]
bestParameters = [4,65,0.2]

#bestParameters = [3,50,0.0001]
#bestArchitecture = [4,1024,0.5]

for i in range(1,len(sys.argv)):
    testSamples,testScores = readDataset(sys.argv[i])
    trainSamples,trainScores = readAllDatasets()
    network = trainNet(trainSamples,trainScores,bestArchitecture,bestParameters)
#    predictions,MSE = runNet(testSamples,testScores,network)
    predictions,MSE = runNet(trainSamples[:int(len(trainSamples)*0.1)],trainScores[:int(len(trainScores)*0.1)],network)
#   printStatistics(predictions,testScores,MSE)
    printStatistics(predictions,trainScores,MSE)
    predictions,MSE = runNet(testSamples,testScores,network)
    printStatistics(predictions,testScores,MSE)
