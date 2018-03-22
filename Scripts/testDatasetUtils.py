import numpy as np
from definitions import *

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

def trainNet(trainData,trainScores,architecture,parameters): 

    records = []
    numLayers,initialNeurons,neuronDecay = architecture
    numLayers,initialNeurons,neuronDecay = [int(numLayers),int(initialNeurons),round(neuronDecay,2)]

    epochs,batch_size,sigma = parameters
    epochs,batch_size,sigma = [int(epochs),int(batch_size),sigma]
    
    if trainScores.shape[1]: 
        numOutputs = 2
    else: 
        numOuputs = 1
    opt,out,mse,X,Y = defineNet(numLayers,initialNeurons,neuronDecay,sigma,len(trainData[0]),numOutputs)
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
            if i % 100 == 0:
                print "MSE for iter "+str(i)+" of "+str(len(trainScores) // batch_size)+" is :"\
                    +str(round(net.run(mse, feed_dict={X: batch_x, Y: batch_y}),4))
    network = [net,out,mse,X,Y]
    return network

def runNet(testData,testScores,network): 
    net,out,mse,X,Y = network
    MSE = round(net.run(mse, feed_dict={X: testData, Y: testScores}),4)
    predictions = net.run(out, feed_dict={X: testData})
    return predictions,MSE

def printStatistics(predictions, testScores, MSE): 
    numCorrect = 0
    numFailedHorribly = 0
    print "Predicted scores vs actual scores : "
    if len(predictions[0]) == 1: 
        for i in range(0,len(predictions)): 
            diffHomeScore = abs(float(predictions[i])-float(testScores[i]))/(float(testScores[i])+1)
            diffVisitorScore = abs(float(predictions[i])-float(testScores[i]))/(float(testScores[i])+1)
            if diffHomeScore+diffVisitorScore/2 <= 0.2: 
                numCorrect = numCorrect + 1 
                print "\t"+str(round(predictions[i],2))+" \t"+str(testScores[i])+" *"
            elif diffHomeScore+diffVisitorScore/2 >= 0.5:
                numFailedHorribly = numFailedHorribly + 1
                print "\t"+str(round(predictions[i],2))+" \t"+str(testScores[i])+" x"
            else: 
                print "\t"+str(round(predictions[i],2))+" \t"+str(testScores[i])
    else: 
        for i in range(0,len(predictions)): 
            diffHomeScore = abs(float(predictions[i][0])-float(testScores[i][0]))/(float(testScores[i][0])+1)
            diffVisitorScore = abs(float(predictions[i][1])-float(testScores[i][1]))/(float(testScores[i][1])+1)
            if diffHomeScore+diffVisitorScore/2 <= 0.2:
                numCorrect = numCorrect + 1
                print"\t"+str(round(predictions[i][0],2))+","+str(round(predictions[i][1],2))+"\t"+str(testScores[i][0])+","+str(testScores[i][1])+" *"
            elif diffHomeScore+diffVisitorScore/2 >= 0.5:
                numFailedHorribly = numFailedHorribly + 1
                print"\t"+str(round(predictions[i][0],2))+","+str(round(predictions[i][1],2))+"\t"+str(testScores[i][0])+","+str(testScores[i][1])+" x"
            else:
                print"\t"+str(round(predictions[i][0],2))+","+str(round(predictions[i][1],2))+"\t"+str(testScores[i][0])+","+str(testScores[i][1])


    print "Number within 20% of actual score : "+str(numCorrect)+" of "+str(len(predictions))+" predictions"
    print "Percent within 20% of actual score : "+str(float(numCorrect)/len(predictions)*100)
    print "Number failed horribly : "+str(numFailedHorribly)+" of "+str(len(predictions))+" predictions"
    print "Percent failed horribly : "+str(float(numFailedHorribly)/len(predictions)*100)
    print "Final MSE : "+str(MSE)
