import tensorflow as tf
import re
import numpy as np
from utils import * 
from definitions import *

allSamples,allScores,dates = readAllDatasets()
trainData,testData,trainScores,testScores = divideData(allSamples,allScores)

iterations = 4
bestArchitecture = [2,1024,0.5]
bestParameters = [3,25,0.1]
history = []
for i in range(0,iterations): 
    [bestParameters,bestMSE,bestError] = findBestParameters(trainData,trainScores,testData,testScores,bestParameters,bestArchitecture)
    history.append([bestParameters,bestArchitecture,bestMSE,bestError])
    [bestArchitecture,bestMSE,bestError] = findBestArchitecture(trainData,trainScores,testData,testScores,bestParameters,bestArchitecture)
    history.append([bestParameters,bestArchitecture,bestMSE,bestError])
    print "\n-----------------------------------------------------------------------------------------------------------------\n"
    print "Best parameters = "+str(bestParameters)+" and best architecture = "+str(bestArchitecture)+" for iteration "+str(i),\
            " with MSE : "+str(bestMSE)+" and error : "+str(bestError)
    print "\n-----------------------------------------------------------------------------------------------------------------\n"

printGeneticHistory(history)

