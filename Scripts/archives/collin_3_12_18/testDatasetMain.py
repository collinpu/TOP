import tensorflow as tf
import re
import numpy as np
from testDatasetUtils import * 
from definitions import *
import sys

#bestParameters = [11,5,0.00001]
#bestArchitecture = [4,1024,0.77]

bestParameters = [3,50,0.00001]
bestArchitecture = [4,1024,0.77]

for i in range(1,len(sys.argv)):
    testSamples,testScores = readDataset(sys.argv[i])
    trainSamples,trainScores = readAllDatasets()
    network = trainNet(trainSamples,trainScores,bestArchitecture,bestParameters)
    predictions = runNet(testSamples,network)
    printStatistics(predictions,testScores)
