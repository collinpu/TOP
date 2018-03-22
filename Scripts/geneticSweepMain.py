import tensorflow as tf
import re
import numpy as np
from dataUtils import * 
from geneticSweepDefinitions import *
from printUtils import *

allSamples,allScores,nameDictionary = readAllDatasets()
sweepData = divideData(allSamples,allScores)

testData = readDataset('../Datasets/USC2017.csv',nameDictionary)

iterations = 100
#bestArchitecture = [5,1512,0.5]
#bestParameters = [3,25,0.0001]

bestSweepParameters = [1,200,5,128,0.25,0.0001]    # [epochs,batchSize,numLayers,initialNeurons,layerDecay,sigma]
bestSweepPerformance = [float('inf'),float('inf')]
geneticHistory = []
performanceHistory = []
sweepTypes = ['RunParameters','ArchitectureParameters']
for i in range(0,iterations): 
    for sweepType in sweepTypes: 
        [newSweepParameters,newSweepPerformance] = parameterSweep(sweepData,bestSweepParameters,bestSweepPerformance,sweepType)
        if newSweepPerformance[0] < bestSweepPerformance[0]:
            bestSweepParameters = newSweepParameters
            bestSweepPerformance = newSweepPerformance
            updateBestModel(sweepType)

    print "\n-----------------------------------------------------------------------------------------------------------------\n"
    print "Best run parameters = "+str(bestSweepParameters[:2])+" and best architecture = "+str(bestSweepParameters[2:])+" for iteration "+str(i),\
            " with MSE : "+str(bestSweepPerformance[0])+" and error : "+str(bestSweepPerformance[1])
    print "\n-----------------------------------------------------------------------------------------------------------------\n"

    geneticHistory.append([bestSweepParameters,bestSweepPerformance]) 
    printGeneticHistory(geneticHistory)
    
    predictions = runNet(testData,bestSweepParameters)
    printStatistics(predictions,testData,performanceHistory,geneticHistory)

