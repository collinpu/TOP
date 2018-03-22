import tensorflow as tf
import re
import numpy as np
from geneticUtils import * 
from definitions import *

allSamples,allScores = readAllDatasets()
trainData,testData,trainScores,testScores = divideData(allSamples,allScores)

iterations = 100
bestArchitecture = [5,1512,0.5]
bestParameters = [3,25,0.0001]
#bestArchitecture = [1,10,1]
#bestParameters = [3,50,0.0001]
#bestArchitecture = [5,128,0.25]
#bestParameters = [1,200,0.2]

bestMSE = float('inf')
bestError = float('inf')
history = []
performanceHistory = []
for i in range(0,iterations): 
    [newParameters,newMSE,newError] = findBestParameters(trainData,trainScores,testData,testScores,bestParameters,bestArchitecture,bestMSE,bestError)
    if newMSE < bestMSE: 
        bestParameters = newParameters
        bestMSE = newMSE
        bestError = newError
        bestTrial = "Parameter"
    history.append([bestParameters,bestArchitecture,bestMSE,bestError])
    
    [newArchitecture,newMSE,newError]=findBestArchitecture(trainData,trainScores,testData,testScores,bestParameters,bestArchitecture,bestMSE,bestError)
    if newMSE < bestMSE: 
        bestArchitecture = newArchitecture
        bestMSE = newMSE
        bestError = newError
        bestTrial = "Architecture"
    history.append([bestParameters,bestArchitecture,bestMSE,bestError])
    print "\n-----------------------------------------------------------------------------------------------------------------\n"
    print "Best parameters = "+str(bestParameters)+" and best architecture = "+str(bestArchitecture)+" for iteration "+str(i),\
            " with MSE : "+str(bestMSE)+" and error : "+str(bestError)
    print "\n-----------------------------------------------------------------------------------------------------------------\n"

    testSamples,testLabels = readDataset('../Datasets/USC2017.csv')
    print bestParameters
    print bestArchitecture
    print bestTrial
    predictions = runNet(bestParameters,bestArchitecture,testSamples,testLabels,bestTrial)
    printStatistics(predictions,testScores,performanceHistory)
    printPerformanceHistory(performanceHistory)
    printGeneticHistory(history)

