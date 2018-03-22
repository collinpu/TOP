import numpy as np
from definitions import *

def readAllDatasets():
    lenVisTeamDimentions = 17
    dates = []
    visitorDimentions = []
    homeDimentions = []
    files = ['2010.csv','2011.csv','2012.csv','2013.csv','2014.csv','2015.csv','2016.csv']
    files = ['../Datasets/'+x for x in files]

    for i in range(0,len(files)): 
        samples = [line.strip('\n').strip(' ').split(',') for line in open(files[i])]
        for j in range(1,len(samples)):
            dates.append(samples[j][0])
            visitorDimentions.append(samples[j][1:lenVisTeamDimentions+1])
            homeDimentions.append(samples[j][lenVisTeamDimentions+1:])

    visitorScores = np.asarray(visitorDimentions)[:,-1]
    visitorDimentions = np.delete(np.asarray(visitorDimentions),np.s_[::len(visitorDimentions[0])-1], 1)

    homeScores = np.asarray(homeDimentions)[:,-1]
    homeDimentions = np.delete(np.asarray(homeDimentions),np.s_[::len(homeDimentions[0])-1],1)

    homeSamples = np.append(homeDimentions,visitorDimentions,1)
    visitorSamples = np.append(visitorDimentions,homeDimentions,1)
    allSamples = np.append(homeSamples,visitorSamples,0)
    allScores = np.append(homeScores,visitorScores)
    allDates = np.append(dates,dates)
 
    return allSamples,allScores,allDates

def divideData(allSamples,allScores): 
    trainingSize = int(0.7*len(allSamples))

    trainData = allSamples[:trainingSize]
    testData  = allSamples[trainingSize+1:]

    trainScores = allScores[:trainingSize]
    testScores  = allScores[trainingSize+1:]

    return trainData,testData,trainScores,testScores

def generateParameterPermutations(bestParameters): 
    numEpochs = 2
    numBatchSizes = 2
    numSigmas = 2
    epochs,batchSize,sigma = bestParameters
    epochs,batchSize,sigma = [int(epochs),int(batchSize),sigma]
    epochPermutations = np.linspace(max(epochs-2,1),epochs+2,numEpochs)
    batchSizePermutations = np.linspace(max(batchSize-5,1),batchSize+5,numBatchSizes)
    sigmaPermutations = np.linspace(sigma/10,sigma*10,numSigmas)
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
    layers,initialNeurons,nodeDecay = bestArchitecture
    layers,initialNeurons,nodeDecay = [int(layers),int(initialNeurons),round(nodeDecay,2)]
    layerPermutations = np.linspace(max(layers-1,1),layers+1,numLayerPermutations)
    intialNeuronPermutations = np.linspace(max(initialNeurons-512,128),initialNeurons+512,numInitialNeuronPermutations)
    nodeDecayPermutations = np.linspace(max(nodeDecay-0.125,0.125),nodeDecay+0.125,numNodeDecayPermutations) 
    
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

    f = open("output.txt","w")
    f.writelines(str(bestMSEs),str(bestErrors),str(bestEpochs),str(bestBatchSizes),str(bestSigmas),\
                    str(bestLayers),str(bestInitialNeurons),str(bestNodeDecays))
    f.close
def findBestParameters(trainData,trainScores,testData,testScores,bestParameters,bestArchitecture): 
    parameterPermutations = generateParameterPermutations(bestParameters)
    bestMSE = float('inf')
    records = []
    epochDictionary = {}
    batchSizeDictionary = {}
    sigmaDictionary = {}
    numLayers,initialNeurons,neuronDecay = bestArchitecture
    numLayers,initialNeurons,neuronDecay = [int(numLayers),int(initialNeurons),round(neuronDecay,2)]
    for permutation in parameterPermutations:
        epochs,batch_size,sigma = permutation
        [epochs,batch_size,sigma] = [int(epochs),int(batch_size),sigma]

        opt,out,mse,X,Y = defineNet(numLayers,initialNeurons,neuronDecay,sigma,len(trainData[0]))
        net = tf.Session()
        net.run(tf.global_variables_initializer())

        print "Run for permutation with "+str(epochs)+" epochs, "+str(batch_size)+" batch size, and "+str(sigma)+" sigma value"
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
                    predictions = [float(x) for x in predictions[0]]
                    testScores = [float(x) for x in testScores]
                    difference = np.subtract(predictions,testScores)
                    error = np.linalg.norm(difference)
                    print "Error at epoch "+str(e)+" iteratiion number "+str(i)+" is: "+str(round(error,2))

        mse_final = round(net.run(mse, feed_dict={X: testData, Y: testScores}),4)
        print "Final MSE for "+str(epochs)+" epochs, "+str(batch_size)+" batch size, and "+str(sigma)+" sigma is :"+str(mse_final)
        predictions = net.run(out, feed_dict={X: testData})
        predictions = [float(x) for x in predictions[0]]
        testScores = [float(x) for x in testScores]
        difference = np.subtract(predictions,testScores)
        error = round(np.linalg.norm(difference),2)

        epochDictionary.setdefault('epoch'+str(epochs),[]).append(mse_final)
        batchSizeDictionary.setdefault('batchSize'+str(batch_size),[]).append(mse_final)
        sigmaDictionary.setdefault('sigma'+str(sigma),[]).append(mse_final)

        records.append([permutation,error,mse_final])
        if mse_final < bestMSE:
            bestMSE = mse_final
            bestPermutation = [epochs,batch_size,sigma]
            bestError = error
    bestTrial = [bestPermutation,bestMSE,bestError]
    printParameterStatistics(bestTrial,records,epochDictionary,batchSizeDictionary,sigmaDictionary)
    return bestTrial

def findBestArchitecture(trainData,trainScores,testData,testScores,bestParameters,bestArchitecture): 
    networkPermutations = generateNetworkPermutations(bestArchitecture)
    bestMSE = float('inf')
    records = []
    layerDictionary = {}
    initialNeuronsDictionary = {} 
    layerDecayDictionary = {}
    epochs,batch_size,sigma = bestParameters
    epochs,batch_size,sigma = [int(epochs),int(batch_size),sigma]
    print epochs,batch_size,sigma
    for permutation in networkPermutations:
        numLayers,initialNeurons,layerDecay = permutation
        numLayers,initialNeurons,layerDecay = [int(numLayers),int(initialNeurons),round(layerDecay,2)]

        opt,out,mse,X,Y = defineNet(numLayers,initialNeurons,layerDecay,sigma,len(trainData[0]))
        net = tf.Session()
        net.run(tf.global_variables_initializer())

        print "Run for permutation with "+str(numLayers)+" layers, "+str(initialNeurons)+" initial neurons, and "+str(layerDecay)+" layer decay"
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
                    predictions = [float(x) for x in predictions[0]]
                    testScores = [float(x) for x in testScores]
                    difference = np.subtract(predictions,testScores)
                    error = np.linalg.norm(difference)
                    print "Error at epoch "+str(e)+" iteratiion number "+str(i)+" is: "+str(round(error,2))

        mse_final = round(net.run(mse, feed_dict={X: testData, Y: testScores}),4)
        print "Final MSE for "+str(numLayers)+" layers, "+str(initialNeurons)+" initial layer, and "+str(layerDecay)+" layerDecay is :"+str(mse_final)
        predictions = net.run(out, feed_dict={X: testData})
        predictions = [float(x) for x in predictions[0]]
        testScores = [float(x) for x in testScores]
        difference = np.subtract(predictions,testScores)
        error = round(np.linalg.norm(difference),2)

        layerDictionary.setdefault(str(numLayers),[]).append(mse_final)
        initialNeuronsDictionary.setdefault(str(initialNeurons),[]).append(mse_final)
        layerDecayDictionary.setdefault(str(layerDecay),[]).append(mse_final)

        records.append([permutation,error,mse_final])
        if mse_final < bestMSE:
            bestMSE = mse_final
            bestPermutation = [numLayers,initialNeurons,layerDecay]
            bestError = error
    bestTrial = [bestPermutation,bestMSE,bestError]
    printArchitectureStatistics(bestTrial,records,layerDictionary,initialNeuronsDictionary,layerDecayDictionary)
    return bestTrial

