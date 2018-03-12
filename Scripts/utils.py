import numpy as np
def readAllDatasets():
    lenVisTeamDimentions = 17
    date = []
    visitorSamples = []
    homeSamples = []
    files = ['2010.csv','2011.csv','2012.csv','2013.csv','2014.csv','2015.csv','2016.csv']
    files = ['../Datasets/'+x for x in files]

    for i in range(0,len(files)): 
        samples = [line.strip('\n').strip(' ').split(',') for line in open(files[i])]
        for j in range(1,len(samples)):
            date.append(samples[j][0])
            visitorSamples.append(samples[j][1:lenVisTeamDimentions+1])
            homeSamples.append(samples[j][lenVisTeamDimentions+1:])

    visitorScores = np.asarray(visitorSamples)[:,-1]
    visitorSamples = np.delete(np.asarray(visitorSamples),np.s_[::len(visitorSamples[0])-1], 1)

    homeScores = np.asarray(homeSamples)[:,-1]
    homeSamples = np.delete(np.asarray(homeSamples),np.s_[::len(homeSamples[0])-1],1)

    return visitorSamples, homeSamples, visitorScores, homeScores, date

def generateParameterPermutations(): 
    numEpochs = 3
    numBatchSizes = 3
    numSigmas = 10
    epochPermutations = np.linspace(3,10,numEpochs)
    batchSizePermutations = np.linspace(5,50,numBatchSizes)
    sigmaPermutations = np.linspace(0.001,1,numSigmas)
    permutations = []
    for i in range(0,numEpochs):
        for j in range(0,numBatchSizes): 
            for k in range(0,numSigmas): 
                permutations.append([epochPermutations[i],batchSizePermutations[j],sigmaPermutations[k]])
    return np.asarray(permutations)
               
def printStatistics(bestTrial,records,epochDictionary,batchSizeDictionary,sigmaDictionary): 
    bestPermutation,bestMSE,bestError = bestTrial
    epochs,batch_size,sigma = bestPermutation
    print "\n-------------------------------------------------------------------------"
    print("Best permutation is "+str(epochs)+" epochs, "+str(batch_size)+" batch size, and "+str(sigma)+" sigma with an MSE of "+str(bestMSE),\
                " and error "+str(bestError))
    print "\nAverage MSE for each epoch permutation"
    for numEpochs,MSEs in epochDictionary.iteritems(): 
        print "\t"+str(numEpochs)+"\t : \t"+str(np.mean(MSEs))
    print "\nAverage MSE for each batch size permutation"
    for batchSize,MSEs in batchSizeDictionary.iteritems(): 
        print "\t"+str(batchSize)+"\t : \t"+str(np.mean(MSEs))
    print "\nAverage MSE for each sigma permutation"
    for sigma,MSEs in sigmaDictionary.iteritems(): 
        print "\t"+str(sigma)+"\t : \t"+str(np.mean(MSEs))

    print "\nRecord of runs:"
    for x in records:
        print("Permutation : ["+str(x[0][0])+","+str(x[0][1])+","+str(x[0][2])+"] \tError : "+str(x[1])+" \tMSE : "+str(x[2]))

