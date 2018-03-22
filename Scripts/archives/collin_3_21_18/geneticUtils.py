import numpy as np
from definitions import *
from psutil import * 
import os

def normilize(data):
    mins = np.amin(data,0)
    maxs = np.amax(data,0)
    normilized = [[float((x[i]-mins[i])/(maxs[i]-mins[i])) for i in range(0,len(x))] for x in data]
    return np.asarray(normilized)

def preProcessData(visitorDimentions,homeDimentions,removeMedianData,normilizeData):
    visitorScores = np.asarray(visitorDimentions)[:,-1]
    visitorDimentions = np.delete(np.asarray(visitorDimentions),np.s_[::len(visitorDimentions[0])-1], 1)

    homeScores = np.asarray(homeDimentions)[:,-1]
    homeDimentions = np.delete(np.asarray(homeDimentions),np.s_[::len(homeDimentions[0])-1],1)

    homeSamples = np.append(homeDimentions,visitorDimentions,1)
    visitorSamples = np.append(visitorDimentions,homeDimentions,1)
    allSamples = np.asarray(np.append(homeSamples,visitorSamples,0),np.float32)

    numScores = 2
    indices = []
    if numScores == 2:
        homeScores_ = np.append(np.transpose([homeScores]),np.transpose([visitorScores]),1)
        visitorScores_ = np.append(np.transpose([visitorScores]),np.transpose([homeScores]),1)
        allScores = np.append(homeScores_,visitorScores_,0)
        if removeMedianData:
            for i in range(0,len(allScores)):
                if int(allScores[i][0]) < 35 and int(allScores[i][0]) > 25:
                    indices.append(i)
    else:
        allScores = np.append(homeScores,visitorScores,0)
        if removeMedianData:
            for i in range(0,len(allScores)):
                if int(allScores[i]) < 35 and int(allScores[i]) > 25:
                    indices.append(i)

    allScores = np.delete(allScores,indices,0)
    allSamples = np.delete(allSamples,indices,0)
    if normilizeData:
        allSamples = normilize(allSamples)
    return allSamples,allScores

def readDataset(fileName):
    lenVisTeamDimentions = 17
    visitorDimentions = [] 
    homeDimentions = []
    samples = [line.strip('\n').strip(' ').split(',') for line in open(fileName)]
    for j in range(1,len(samples)):
        visitorDimentions.append(samples[j][1:lenVisTeamDimentions+1])
        homeDimentions.append(samples[j][lenVisTeamDimentions+1:])
    removeMedianData = False
    normilizeData = False
    return preProcessData(homeDimentions,visitorDimentions,removeMedianData,normilizeData)

def readAllDatasets():
    lenVisTeamDimentions = 17
    visitorDimentions = []
    homeDimentions = []
    files = ['2009.csv','2010.csv','2011.csv','2012.csv','2013.csv','2014.csv','2015.csv','2016.csv']
    files = ['../Datasets/'+x for x in files]
    for i in range(0,len(files)):
        samples = [line.strip('\n').strip(' ').split(',') for line in open(files[i])]
        for j in range(1,len(samples)):
            visitorDimentions.append(samples[j][1:lenVisTeamDimentions+1])
            homeDimentions.append(samples[j][lenVisTeamDimentions+1:])
    removeMedianData = False
    normilizeData = False
    return preProcessData(homeDimentions,visitorDimentions,removeMedianData,normilizeData)

def divideData(allSamples,allScores): 
    trainingSize = int(0.9*len(allSamples))

    trainData = allSamples[:trainingSize]
    testData  = allSamples[trainingSize+1:]

    trainScores = allScores[:trainingSize]
    testScores  = allScores[trainingSize+1:]

    return trainData,testData,trainScores,testScores

def generateParameterPermutations(bestParameters): 
    numEpochs = 2
    numBatchSizes = 2
    numSigmas = 2
    epochChange = np.random.randint(1,4,1)
    batchChange = np.random.randint(1,4,1)*5
    sigmaChange = np.random.randint(1,2,1)*10
    epochs,batchSize,sigma = bestParameters
    epochs,batchSize,sigma = [int(epochs),int(batchSize),sigma]
    epochPermutations = np.linspace(max(epochs-epochChange,1),epochs+epochChange,numEpochs)
    batchSizePermutations = np.linspace(max(batchSize-batchChange,1),batchSize+batchChange,numBatchSizes)
    sigmaPermutations = np.linspace(sigma/sigmaChange,min(10,sigma*sigmaChange,numSigmas))
    permutations = []
    for i in range(0,numEpochs):
        for j in range(0,numBatchSizes): 
            for k in range(0,numSigmas):
                permutations.append([epochPermutations[i],batchSizePermutations[j],sigmaPermutations[k]])
    return np.asarray(permutations)
              
def generateNetworkPermutations(bestArchitecture):
    numLayerPermutations = 2
    numInitialNeuronPermutations = 2
    numNodeDecayPermutations = 2
    layerChange = np.random.randint(1,3,1)
    neuronChange = np.random.randint(1,5,1)*128
    decayChange = np.random.randint(1,3,1)*0.125
    layers,initialNeurons,nodeDecay = bestArchitecture
    layers,initialNeurons,nodeDecay = [int(layers),int(initialNeurons),round(nodeDecay,2)]
    layerPermutations = np.linspace(max(layers-layerChange,1),layers+layerChange,numLayerPermutations)
    intialNeuronPermutations = np.linspace(max(initialNeurons-neuronChange,128),initialNeurons+neuronChange,numInitialNeuronPermutations)
    nodeDecayPermutations = np.linspace(max(nodeDecay-decayChange,0.125),min(nodeDecay+decayChange,1.25),numNodeDecayPermutations) 
    
    permutations = [] 
    for i in range(0,numLayerPermutations):
        for j in range(0,numInitialNeuronPermutations): 
            for k in range(0,numNodeDecayPermutations): 
                permutations.append([layerPermutations[i],intialNeuronPermutations[j],nodeDecayPermutations[k]])
    return permutations

def printParameterStatistics(bestTrial,records,epochDictionary,batchSizeDictionary,sigmaDictionary): 
    bestPermutation,bestMSE,bestError = bestTrial
    epochs,batch_size,sigma = bestPermutation
    print "\n-------------------------------------------------------------------------"
    print("Best permutation is "+str(epochs)+" epochs, "+str(batch_size)+" batch size, and "+str(sigma)+" sigma with an MSE of "+str(bestMSE),\
                " and error "+str(bestError))
    print "\nAverage MSE for each epoch permutation"
    for numEpochs,MSEs in epochDictionary.iteritems(): 
        print "\t"+str(numEpochs)+" \t :  \t"+str(np.mean(MSEs))
    print "\nAverage MSE for each batch size permutation"
    for batchSize,MSEs in batchSizeDictionary.iteritems(): 
        print "\t"+str(batchSize)+" \t :  \t"+str(np.mean(MSEs))
    print "\nAverage MSE for each sigma permutation"
    for sigma,MSEs in sigmaDictionary.iteritems(): 
        print "\t"+str(sigma)+" \t :  \t"+str(np.mean(MSEs))

    print "\nRecord of runs:"
    for x in records:
        print("Permutation : ["+str(x[0][0])+","+str(x[0][1])+","+str(x[0][2])+"]   \tError : "+str(x[1])+"  \tMSE : "+str(x[2]))

def printArchitectureStatistics(bestTrial,records,layerDictionary,initialNeuronsDictionary,layerDecayDictionary):
    bestPermutation,bestMSE,bestError = bestTrial
    numLayers,initialNeurons,layerDecay = bestPermutation
    print "\n-------------------------------------------------------------------------"
    print("Best permutation is "+str(numLayers)+" layers, "+str(initialNeurons)+" initial neurons, and "+str(layerDecay),\
        " layer decay with an MSE of "+str(bestMSE)+" and error "+str(bestError))
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
        bestSigmas.append(generation[0][2])
        bestLayers.append(int(generation[1][0]))
        bestInitialNeurons.append(int(generation[1][1]))
        bestNodeDecays.append(generation[1][2])
        bestMSEs.append(round(generation[2],4))
        bestErrors.append(round(generation[3],2))

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

    print "Writing to output.txt"
    f = open("output.txt","w")
    f.writelines([str(bestMSEs),str(bestErrors),str(bestEpochs),str(bestBatchSizes),str(bestSigmas),\
                    str(bestLayers),str(bestInitialNeurons),str(bestNodeDecays)])
    f.close
    print "Done"
def findBestParameters(trainData,trainScores,testData,testScores,bestParameters,bestArchitecture,bestMSE,bestError): 
    parameterPermutations = generateParameterPermutations(bestParameters)
    bestPermutation = bestParameters
    iterationNum = 1
    records = []
    epochDictionary = {}
    batchSizeDictionary = {}
    sigmaDictionary = {}
    numLayers,initialNeurons,neuronDecay = bestArchitecture
    numLayers,initialNeurons,neuronDecay = [int(numLayers),int(initialNeurons),round(neuronDecay,2)]
    for permutation in parameterPermutations:
        epochs,batch_size,sigma = permutation
        [epochs,batch_size,sigma] = [int(epochs),int(batch_size),sigma]

        print "Run for permutation with "+str(epochs)+" epochs, "+str(batch_size)+" batch size, and "+str(sigma)+" sigma value"
        print "\twith "+str(numLayers)+" layers, "+str(initialNeurons)+" initial neurons, and "+str(neuronDecay)+" neuron decay"
        print "Parameter permutation number "+str(iterationNum)+" of "+str(len(parameterPermutations))
        g = tf.Graph()
        net = tf.InteractiveSession(graph=g)
        with g.as_default(): 
            opt,out,mse,X,Y = defineNet(numLayers,initialNeurons,neuronDecay,sigma,len(trainData[0]),len(trainScores[0]))
            saver = tf.train.Saver()
            net.run(tf.global_variables_initializer())
            percentMemoryUsed = virtual_memory()[2]
            print("Percent memory used : "+str(percentMemoryUsed))
            for e in range(0,epochs):
                # Shuffle training data
                shuffle_indices = np.random.permutation(np.arange(len(trainData)))
                trainData = trainData[shuffle_indices]
                trainScores = trainScores[shuffle_indices]
                # Minibatch training
                for i in range(0, len(trainScores) // batch_size):
                    start = i * int(batch_size)
                    batch_x = trainData[start:start + batch_size]
                    batch_y = trainScores[start:start + batch_size]
                    # Run optimizer with batch
                    net.run(opt, feed_dict={X: batch_x, Y: batch_y})

                    # Show progress
                    if np.mod(i, 100) == 0:
                        # Prediction
                        predictions = net.run(out, feed_dict={X: testData})
                        predictions = [[float(x) for x in y] for y in predictions]
                        testScores = [[float(x) for x in y] for y in testScores]
                        error = round(np.linalg.norm(np.subtract(predictions,testScores)),2)
                        print"Error at epoch "+str(e+1)+" iteratiion number "+str(i)+" of "+str(len(trainScores)//batch_size)+" is: "+str(error)
                
            mse_final = round(net.run(mse, feed_dict={X: testData, Y: testScores}),4)
            print "Final MSE for "+str(epochs)+" epochs, "+str(batch_size)+" batch size, and "+str(sigma)+" sigma is :"+str(mse_final)
            predictions = net.run(out, feed_dict={X: testData})
            predictions = [[float(x) for x in y] for y in predictions]
            testScores = [[float(x) for x in y] for y in testScores]
            error = round(np.linalg.norm(np.subtract(predictions,testScores)),2)

            epochDictionary.setdefault('epoch'+str(epochs),[]).append(mse_final)
            batchSizeDictionary.setdefault('batchSize'+str(batch_size),[]).append(mse_final)
            sigmaDictionary.setdefault('sigma'+str(sigma),[]).append(mse_final)

            records.append([permutation,error,mse_final])
        if mse_final < bestMSE:
            bestMSE = mse_final
            bestPermutation = [epochs,batch_size,sigma]
            bestError = error
            saver.save(net, "tmp/bestParameterModel.ckpt")

        net.close()
        tf.reset_default_graph()
        iterationNum = iterationNum + 1
    bestTrial = [bestPermutation,bestMSE,bestError]
    printParameterStatistics(bestTrial,records,epochDictionary,batchSizeDictionary,sigmaDictionary)
    return bestTrial

def findBestArchitecture(trainData,trainScores,testData,testScores,bestParameters,bestArchitecture,bestMSE,bestError): 
    networkPermutations = generateNetworkPermutations(bestArchitecture)
    bestPermutation = bestArchitecture
    iterationNum = 1
    records = []
    layerDictionary = {}
    initialNeuronsDictionary = {} 
    layerDecayDictionary = {}
    epochs,batch_size,sigma = bestParameters
    epochs,batch_size,sigma = [int(epochs),int(batch_size),sigma]
    for permutation in networkPermutations:
        print "Architecture permutation number "+str(iterationNum)+" of "+str(len(networkPermutations))

        numLayers,initialNeurons,layerDecay = permutation
        numLayers,initialNeurons,layerDecay = [int(numLayers),int(initialNeurons),round(layerDecay,2)]
        g = tf.Graph()
        net = tf.InteractiveSession(graph=g)
        with g.as_default():
            opt,out,mse,X,Y = defineNet(numLayers,initialNeurons,layerDecay,sigma,len(trainData[0]),len(trainScores[0]))
            saver = tf.train.Saver()
            net.run(tf.global_variables_initializer())
            print "Run for permutation with "+str(epochs)+" epochs, "+str(batch_size)+" batch size, and "+str(sigma)+" sigma value"
            print "\twith "+str(numLayers)+" layers, "+str(initialNeurons)+" initial neurons, and "+str(layerDecay)+" neuron decay"
            percentMemoryUsed = virtual_memory()[2]
            print("Percent memory used : "+str(percentMemoryUsed))
            for e in range(0,epochs):
                # Shuffle training data
                shuffle_indices = np.random.permutation(np.arange(len(trainData)))
                trainData = trainData[shuffle_indices]
                trainScores = trainScores[shuffle_indices]

                # Minibatch training
                for i in range(0, len(trainScores) // batch_size):
                    start = i * int(batch_size)
                    batch_x = trainData[start:start + batch_size]
                    batch_y = trainScores[start:start + batch_size]
                    # Run optimizer with batch
                    net.run(opt, feed_dict={X: batch_x, Y: batch_y})

                    # Show progress
                    if np.mod(i, 100) == 0:
                        # Prediction
                        predictions = net.run(out, feed_dict={X: testData})
                        predictions = [[float(x) for x in y] for y in predictions]
                        testScores = [[float(x) for x in y] for y in  testScores]
                        error = round(np.linalg.norm(np.subtract(predictions,testScores)),2)
                        print"Error at epoch "+str(e+1)+" iteratiion number "+str(i)+" of "+str(len(trainScores)//batch_size)+" is: "+str(error)

            mse_final = round(net.run(mse, feed_dict={X: testData, Y: testScores}),4)
            print"Final MSE for "+str(numLayers)+" layers "+str(initialNeurons)+" initial layer and "+str(layerDecay)+" layerDecay is: "+str(mse_final)
            predictions = net.run(out, feed_dict={X: testData})
            predictions = [[float(x) for x in y] for y in predictions]
            testScores = [[float(x) for x in y] for y in  testScores]
            error = round(np.linalg.norm(np.subtract(predictions,testScores)),2)

            layerDictionary.setdefault(str(numLayers),[]).append(mse_final)
            initialNeuronsDictionary.setdefault(str(initialNeurons),[]).append(mse_final)
            layerDecayDictionary.setdefault(str(layerDecay),[]).append(mse_final)

            records.append([permutation,error,mse_final])
            
        if mse_final < bestMSE:
            bestMSE = mse_final
            bestPermutation = [numLayers,initialNeurons,layerDecay]
            bestError = error
            saver.save(net, "tmp/bestArchitectureModel.ckpt")

        net.close()
        tf.reset_default_graph()    
        iterationNum = iterationNum + 1 
    bestTrial = [bestPermutation,bestMSE,bestError]
    printArchitectureStatistics(bestTrial,records,layerDictionary,initialNeuronsDictionary,layerDecayDictionary)
    return bestTrial

def runNet(bestParameters,bestArchitecture,testData,testLabels,networkType):
    [numLayers,initialNeurons,neuronDecay] = bestArchitecture
    sigma = bestParameters[2]
    dataDimentions = len(testData[0])
    labelDimentions = len(testLabels[0])
    opt,out,mse,X,Y = defineNet(numLayers,initialNeurons,neuronDecay,sigma,dataDimentions,labelDimentions)
    saver = tf.train.Saver()
    with tf.Session() as net:
        saver.restore(net, "tmp/best"+str(networkType)+"Model.ckpt")
        predictions = net.run(out, feed_dict={X: testData})
    tf.reset_default_graph()
    return predictions


def printStatistics(predictions, testScores, performanceHistory):
    numCorrect = 0
    numFailedHorribly = 0
    print "Predicted scores vs actual scores : "
    for i in range(0,len(predictions)):
        diffHomeScore = abs(float(predictions[i][0])-float(testScores[i][0]))/(float(testScores[i][0])+1)
        diffVisitorScore = abs(float(predictions[i][1])-float(testScores[i][1]))/(float(testScores[i][1])+1)
        if diffHomeScore+diffVisitorScore/2 <= 0.2:
            numCorrect = numCorrect + 1
            print "\t"+str(round(predictions[i][0],2))+","+str(round(predictions[i][1],2))+"  \t"+str(testScores[i][0])+","+str(testScores[i][1])+" *"
        elif diffHomeScore+diffVisitorScore/2 >= 0.5:
            numFailedHorribly = numFailedHorribly + 1
            print "\t"+str(round(predictions[i][0],2))+","+str(round(predictions[i][1],2))+"  \t"+str(testScores[i][0])+","+str(testScores[i][1])+" x"
        else:
            print "\t"+str(round(predictions[i][0],2))+","+str(round(predictions[i][1],2))+"  \t"+str(testScores[i][0])+","+str(testScores[i][1])
    print "Number within 20% of actual score : "+str(numCorrect)+" of "+str(len(predictions))+" predictions"
    print "Percent within 20% of actual score : "+str(float(numCorrect)/len(predictions)*100)
    print "Number failed horribly : "+str(numFailedHorribly)+" of "+str(len(predictions))+" predictions"
    print "Percent failed horribly : "+str(float(numFailedHorribly)/len(predictions)*100)
    performanceHistory.append([numCorrect,numFailedHorribly])
def printPerformanceHistory(performanceHistory): 
    for i in range(0,len(performanceHistory)): 
        [numCorrect,numFailed] = performanceHistory[i]
        print("Performance for iteration "+str(i)+" is "+str(numCorrect)+" within 20% correct and "+str(numFailed)+" wrong by 50% or more")
