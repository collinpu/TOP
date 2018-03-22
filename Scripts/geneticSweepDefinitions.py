import numpy as np
from networkDefinitions import * 
from printUtils import * 
from permutationGenerators import *

def parameterSweep(sweepData,bestParameters,bestPerformance,sweepType):
    [trainData,testData,trainScores,testScores] = sweepData
    epochs,batchSize,numLayers,initialNeurons,layerDecay,sigma = bestParameters
    [epochs,batchSize,numLayers,initialNeurons,layerDecay,sigma] = [int(epochs),int(batchSize),int(numLayers),int(initialNeurons),\
                                                                    round(layerDecay,2),sigma]
    
    parameterPermutations = generateParameterPermutations(bestParameters,sweepType)

    epochDictionary = {}
    batchSizeDictionary = {}
    sigmaDictionary = {}
    layerDictionary = {}
    initialNeuronsDictionary = {}
    layerDecayDictionary = {}
    dictionaries = [epochDictionary,batchSizeDictionary,layerDictionary,initialNeuronsDictionary,layerDecayDictionary,sigmaDictionary]

    bestPermutation = bestParameters
    iterationNum = 1
    records = []
    for permutation in parameterPermutations:
        if sweepType == 'RunParameters': 
            epochs,batchSize = permutation
            [epochs,batchSize] = [int(epochs),int(batchSize)]
        elif sweepType == 'ArchitectureParameters':
            numLayers,initialNeurons,layerDecay,sigma = permutation
            numLayers,initialNeurons,layerDecay,sigma = [int(numLayers),int(initialNeurons),round(layerDecay,2),sigma]
        parameters = [epochs,batchSize,numLayers,initialNeurons,layerDecay,sigma]

        print "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"+str(sweepType)+"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
        print "Run for permutation with "+str(epochs)+" epochs, "+str(batchSize)+" batch size, and "+str(sigma)+" sigma value"
        print "\twith "+str(numLayers)+" layers, "+str(initialNeurons)+" initial neurons, and "+str(layerDecay)+" neuron decay"
        print "Permutation number "+str(iterationNum)+" of "+str(len(parameterPermutations))
        
        g = tf.Graph()
        net = tf.InteractiveSession(graph=g)
        with g.as_default():
            opt,out,mse,X,Y = defineNet(numLayers,initialNeurons,layerDecay,sigma,len(trainData[0]),len(trainScores[0]))
            saver = tf.train.Saver()
            net.run(tf.global_variables_initializer())
            for e in range(0,epochs):
                # Shuffle training data
                shuffle_indices = np.random.permutation(np.arange(len(trainData)))
                trainData = trainData[shuffle_indices]
                trainScores = trainScores[shuffle_indices]

                # Minibatch training
                for i in range(0, len(trainScores) // batchSize):
                    start = i * int(batchSize)
                    batch_x = trainData[start:start + batchSize]
                    batch_y = trainScores[start:start + batchSize]
                    # Run optimizer with batch
                    net.run(opt, feed_dict={X: batch_x, Y: batch_y})

                    # Show progress
                    if np.mod(i, int((len(trainScores) // batchSize)/3)) == 0:
                        # Prediction
                        predictions = net.run(out, feed_dict={X: testData})
                        predictions = [[float(x) for x in y] for y in predictions]
                        testScores = [[float(x) for x in y] for y in testScores]
                        error = round(np.linalg.norm(np.subtract(predictions,testScores)),2)
                        print"Error at epoch "+str(e+1)+" iteratiion number "+str(i)+" of "+str(len(trainScores)//batchSize)+" is: "+str(error)

            finalMSE = round(net.run(mse, feed_dict={X: testData, Y: testScores}),4)
            print("Final MSE for "+str(epochs)+" epochs, "+str(batchSize)+" batch size, "+str(sigma)+" sigma, "\
                    +str(numLayers)+" layers "+str(initialNeurons)+" initial layer and "+str(layerDecay)+" layer decay is: "+str(finalMSE))
            predictions = net.run(out, feed_dict={X: testData})
            predictions = [[float(x) for x in y] for y in predictions]
            testScores = [[float(x) for x in y] for y in testScores]
            error = round(np.linalg.norm(np.subtract(predictions,testScores)),2)
            
            for i in range(0,len(parameters)): 
                dictionaries[i].setdefault(str(parameters[i]),[]).append(finalMSE)
            records.append([permutation,error,finalMSE])

        if finalMSE < bestPerformance[0]:
            bestPerformance = [finalMSE,error]
            bestPermutation = parameters
            if sweepType == 'RunParameters': 
                saver.save(net, "tmp/bestRunParametersModel.ckpt")
            elif sweepType == 'ArchitectureParameters': 
                saver.save(net, "tmp/bestArchitectureParametersModel.ckpt")

        net.close()
        tf.reset_default_graph()
        iterationNum = iterationNum + 1
    bestTrial = [bestPermutation,bestPerformance]
    printParameterStatistics(bestTrial,records,dictionaries,sweepType)
    return bestTrial

def runNet(testData,bestParameters):
    [epochs,batchSize,numLayers,initialNeurons,layerDecay,sigma] = bestParameters
    dataDimentions = len(testData[0][0])
    labelDimentions = len(testData[1][0])
    opt,out,mse,X,Y = defineNet(numLayers,initialNeurons,layerDecay,sigma,dataDimentions,labelDimentions)
    saver = tf.train.Saver()
    with tf.Session() as net:
        saver.restore(net, "tmp/BestModel.ckpt")
        predictions = net.run(out, feed_dict={X: testData[0]})
    tf.reset_default_graph()
    return predictions

