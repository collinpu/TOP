import tensorflow as tf
import re
import numpy as np
from utils import * 
from definitions import *

visitorSamples, homeSamples, visitorScores, homeScores, dates = readAllDatasets()

allSamples = np.append(visitorSamples,homeSamples,0)
allScores = np.append(visitorScores,homeScores,0)

trainingSize = int(0.7*len(allSamples))

trainData = allSamples[:trainingSize]
testData  = allSamples[trainingSize+1:]

trainScores = allScores[:trainingSize]
testScores  = allScores[trainingSize+1:]


parameterPermutations = generateParameterPermutations()
bestMSE = float('inf')
records = []
epochDictionary = {}
batchSizeDictionary = {}
sigmaDictionary = {}
for permutation in parameterPermutations: 
    epochs,batch_size,sigma = permutation

    opt,out,mse,X,Y = defineNet(sigma)
    net = tf.Session()
    net.run(tf.global_variables_initializer())

    print "Run for permutation with "+str(epochs)+" epochs, "+str(batch_size)+" batch size, and "+str(sigma)+" sigma value"
    for e in range(0,int(epochs)):
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(trainData)))
        trainData = trainData[shuffle_indices]
        trainScores = trainScores[shuffle_indices]

        # Minibatch training
        for i in range(0, len(trainScores) // int(batch_size)):
            start = i * int(batch_size)
            batch_x = trainData[start:start + int(batch_size)]
            batch_y = trainScores[start:start + int(batch_size)]
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
                print "Error at epoch "+str(e)+" iteratiion number "+str(i)+" is: "+str(error)

    mse_final = net.run(mse, feed_dict={X: testData, Y: testScores})
    print("Final MSE for "+str(epochs)+" epochs, "+str(batch_size)+" batch size, and "+str(sigma)+" sigma is :", mse_final)
    predictions = net.run(out, feed_dict={X: testData})
    predictions = [float(x) for x in predictions[0]]
    testScores = [float(x) for x in testScores]
    difference = np.subtract(predictions,testScores)
    error = np.linalg.norm(difference)
   
    epochDictionary.setdefault('epoch'+str(epochs),[]).append(mse_final)
    batchSizeDictionary.setdefault('batchSize'+str(batch_size),[]).append(mse_final)
    sigmaDictionary.setdefault('sigma'+str(sigma),[]).append(mse_final)
 
    records.append([permutation,error,mse_final])
    if mse_final < bestMSE: 
        bestMSE = mse_final
        bestPermutation = permutation
        bestError = error
bestTrial = [bestPermutation,bestMSE,bestError]
printStatistics(bestTrial,records,epochDictionary,batchSizeDictionary,sigmaDictionary)

