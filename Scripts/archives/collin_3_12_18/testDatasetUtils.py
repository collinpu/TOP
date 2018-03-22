import numpy as np
from definitions import *

def readDataset(fileName):
    lenVisTeamDimentions = 17
    visitorDimentions = []
    homeDimentions = []

    samples = [line.strip('\n').strip(' ').split(',') for line in open(fileName)]
    for j in range(1,len(samples)):
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
 
    return allSamples,allScores

def readAllDatasets():
    lenVisTeamDimentions = 17
    visitorDimentions = []
    homeDimentions = []
    files = ['2010.csv','2011.csv','2012.csv','2013.csv','2014.csv','2015.csv','2016.csv']
    files = ['../Datasets/'+x for x in files]

    for i in range(0,len(files)):
        samples = [line.strip('\n').strip(' ').split(',') for line in open(files[i])]
        for j in range(1,len(samples)):
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

    return allSamples,allScores


def trainNet(trainData,trainScores,architecture,parameters): 

    records = []
    numLayers,initialNeurons,neuronDecay = architecture
    numLayers,initialNeurons,neuronDecay = [int(numLayers),int(initialNeurons),round(neuronDecay,2)]

    epochs,batch_size,sigma = parameters
    epochs,batch_size,sigma = [int(epochs),int(batch_size),sigma]

    opt,out,mse,X,Y = defineNet(numLayers,initialNeurons,neuronDecay,sigma,len(trainData[0]))
    net = tf.Session()
    net.run(tf.global_variables_initializer())
    print "Running network with : "
    print "\tTraining Parameters  \t : "+str(epochs)+" epochs and "+str(batch_size)+" batch size"
    print "\tNetwork Parameters   \t : "+str(numLayers)+" layers, "+str(initialNeurons)+" initial neurons, "+str(neuronDecay),\
            " neuron decay, and "+str(sigma)+" sigma"
    for e in range(0,epochs):
        print "At epoch "+str(e+1)+" of "+str(epochs)+" epochs"
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(trainData)))
        trainData = trainData[shuffle_indices]
        trainScores = trainScores[shuffle_indices]

        # Minibatch training
        for i in range(0, len(trainScores) // batch_size):
            start = i * batch_size
            batch_x = trainData[start:start + batch_size]
            batch_y = trainScores[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})
    network = [net,out,X]
    return network

def runNet(testData,network): 
    net,out,X = network
    predictions = net.run(out, feed_dict={X: testData})
    return predictions
    
def printStatistics(predictions, testScores): 
    numCorrect = 0
    print "Predicted scores vs actual scores : "
    for i in range(0,len(predictions[0])): 
        print "\t"+str(round(predictions[0][i],2))+"  \t"+str(testScores[i])
        if abs(float(predictions[0][i])-float(testScores[i]))/float(testScores[i]) <= 0.1: 
            numCorrect = numCorrect + 1
    print "Number within 10% of actual score : "+str(numCorrect)+" of "+str(len(predictions[0]))+" predictions"
    print "Percent within 10% of actual score : "+str(numCorrect/len(predictions[0])*100)

