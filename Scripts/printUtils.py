from psutil import *
import numpy as np
from shutil import copyfile
def printParameterStatistics(bestTrial,records,dictionaries,sweepType):
    bestPermutation,bestPerformance = bestTrial
    epochs,batchSize,numLayers,initialNeurons,layerDecay,sigma = bestPermutation
    bestMSE,bestError = bestPerformance
    epochDictionary,batchSizeDictionary,layerDictionary,initialNeuronsDictionary,layerDecayDictionary,sigmaDictionary = dictionaries
    if sweepType == 'RunParameters':
        print "\n-------------------------------------------------------------------------"
        print("Best run permutation is "+str(epochs)+" epochs and "+str(batchSize)+" batch size with an MSE of "+str(bestMSE),\
                    " and error "+str(bestError))
        print "\nAverage MSE for each epoch permutation"
        for numEpochs,MSEs in epochDictionary.iteritems():
            print "\t"+str(numEpochs)+" \t :  \t"+str(np.mean(MSEs))
        print "\nAverage MSE for each batch size permutation"
        for batchSize,MSEs in batchSizeDictionary.iteritems():
            print "\t"+str(batchSize)+" \t :  \t"+str(np.mean(MSEs))
    elif sweepType == 'ArchitectureParameters': 
        print("Best architecture permutation is "+str(numLayers)+" layers, "+str(initialNeurons)+" initial neurons, "+str(layerDecay),\
            " layer decay, and "+str(sigma)+" sigma with an MSE of "+str(bestMSE)+" and error "+str(bestError))
        print "\nAverage MSE for each layer permutation"
        for numLayers,MSEs in layerDictionary.iteritems():
            print "\t"+str(numLayers)+" \t : \t "+str(np.mean(MSEs))
        print "\nAverage MSE for each initial neuron permutation"
        for initialNeurons,MSEs in initialNeuronsDictionary.iteritems():
            print "\t"+str(initialNeurons)+" \t : \t "+str(np.mean(MSEs))
        print "\nAverage MSE for each layer decay permutation"
        for layerDecay,MSEs in layerDecayDictionary.iteritems():
            print "\t"+str(layerDecay)+" \t : \t "+str(np.mean(MSEs))
        for sigma,MSEs in sigmaDictionary.iteritems(): 
            print "\t"+str(sigma)+" \t : \t "+str(np.mean(MSEs))
        print "\nRecord of runs:"
        for x in records:
            print("Permutation : ["+str(x[0][0])+","+str(x[0][1])+","+str(x[0][2])+"]   \tError : "+str(x[1])+"  \tMSE : "+str(x[2]))

def printArchitectureStatistics(bestTrial,records,layerDictionary,initialNeuronsDictionary,layerDecayDictionary):
    bestPermutation,bestMSE,bestError = bestTrial
    numLayers,initialNeurons,layerDecay = bestPermutation
    print "\n-------------------------------------------------------------------------"
    print("Best permutation is "+str(numLayers)+" layers, "+str(initialNeurons)+" initial neurons, "+str(layerDecay),\
        " layer decay, and "+str(sigma)+" sigma with an MSE of "+str(bestMSE)+" and error "+str(bestError))
    print "\nAverage MSE for each layer permutation"
    for numLayers,MSEs in layerDictionary.iteritems():
        print "\t"+str(numLayers)+" \t : \t "+str(np.mean(MSEs))
    print "\nAverage MSE for each initial neuron permutation"
    for initialNeurons,MSEs in initialNeuronsDictionary.iteritems():
        print "\t"+str(initialNeurons)+" \t : \t "+str(np.mean(MSEs))
    print "\nAverage MSE for each layer decay permutation"
    for layerDecay,MSEs in layerDecayDictionary.iteritems():
        print "\t"+str(layerDecay)+" \t : \t "+str(np.mean(MSEs))

    print "\nRecord of runs:"
    for x in records:
        print("Permutation : ["+str(x[0][0])+","+str(x[0][1])+","+str(x[0][2])+"]   \tError : "+str(x[1])+"  \tMSE : "+str(x[2]))
def printGeneticHistory(history):
    bestEpochs = []
    bestBatchSizes = []
    bestSigmas = []
    bestLayers = []
    bestInitialNeurons = []
    bestNodeDecays = []
    bestMSEs = []
    bestErrors = []
    for generation in history:
        bestEpochs.append(int(generation[0][0]))
        bestBatchSizes.append(int(generation[0][1]))
        bestLayers.append(int(generation[0][2]))
        bestInitialNeurons.append(int(generation[0][3]))
        bestNodeDecays.append(generation[0][4])
        bestSigmas.append(generation[0][5])
        bestMSEs.append(round(generation[1][0],0))
        bestErrors.append(round(generation[1][1],1))

    print "\n###########################################################################################"
    print "#                                       FINAL REPORT                                      #"
    print "###########################################################################################\n"

    print "MSE changes            : "+str(bestMSEs)
    print "Error changes          : "+str(bestErrors)
    print "Epoch changes          : "+str(bestEpochs)
    print "Batch Size changes     : "+str(bestBatchSizes)
    print "Sigma changes          : "+str(bestSigmas)
    print "Layers changes         : "+str(bestLayers)
    print "Initial Neuron changes : "+str(bestInitialNeurons)
    print "Node Decay changes     : "+str(bestNodeDecays)
    print "Percent memory used    : "+str(virtual_memory()[2])
    print "Writing to output.txt"
    f = open("tmp/geneticOutput.txt","w")
    f.write(str(bestMSEs)+str(bestErrors)+str(bestEpochs)+str(bestBatchSizes)+str(bestSigmas)\
                    +str(bestLayers)+str(bestInitialNeurons)+str(bestNodeDecays)+"\n")
    f.close

def printStatistics(predictions, testData, performanceHistory,geneticHistory):
    testScores = testData[1]
    numCorrect = 0
    numFailedHorribly = 0
    print "Predicted scores vs actual scores : "
    for i in range(0,len(predictions)):
        diffHomeScore = abs(float(predictions[i][0])-float(testScores[i][0]))/(float(testScores[i][0])+1)
        diffVisitorScore = abs(float(predictions[i][1])-float(testScores[i][1]))/(float(testScores[i][1])+1)
        if diffHomeScore <= 0.2 and diffVisitorScore <= 0.2:
            numCorrect = numCorrect + 1
            print "\t"+str(round(predictions[i][0],2))+","+str(round(predictions[i][1],2))+"  \t"+str(testScores[i][0])+","+str(testScores[i][1])+" *"
        elif diffHomeScore >= 0.5 or diffVisitorScore >= 0.5:
            numFailedHorribly = numFailedHorribly + 1
            print "\t"+str(round(predictions[i][0],2))+","+str(round(predictions[i][1],2))+"  \t"+str(testScores[i][0])+","+str(testScores[i][1])+" x"
        else:
            print "\t"+str(round(predictions[i][0],2))+","+str(round(predictions[i][1],2))+"  \t"+str(testScores[i][0])+","+str(testScores[i][1])
    print "Number within 20% of actual score : "+str(numCorrect)+" of "+str(len(predictions))+" predictions"
    print "Percent within 20% of actual score : "+str(float(numCorrect)/len(predictions)*100)
    print "Number failed horribly : "+str(numFailedHorribly)+" of "+str(len(predictions))+" predictions"
    print "Percent failed horribly : "+str(float(numFailedHorribly)/len(predictions)*100)
    performanceHistory.append([numCorrect,numFailedHorribly])
    f = open("tmp/performanceOutput.txt","w")
    for i in range(0,len(performanceHistory)):
        print("Performance for iteration "+str(i)+" is "+str(performanceHistory[i][0])+" within 20% correct and "+str(performanceHistory[i][1])\
                +" wrong by 50% or more") 
        f.write("For iteration "+str(i)+", [numCorrect,numFailed] : "+str(performanceHistory[i])+" with parameters : "+str(geneticHistory[i])+"\n")
    f.close

def updateBestModel(sweepType): 
    copyfile("tmp/best"+str(sweepType)+"Model.ckpt.data-00000-of-00001","tmp/BestModel.ckpt.data-00000-of-00001")
    copyfile("tmp/best"+str(sweepType)+"Model.ckpt.index","tmp/BestModel.ckpt.index")
    copyfile("tmp/best"+str(sweepType)+"Model.ckpt.meta","tmp/BestModel.ckpt.meta")

