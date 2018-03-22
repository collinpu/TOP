import numpy as np
def normilize(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    return scaler.transform(data)

def hashTeamNames(visitorSamples,homeSamples,nameDictionary):
    if nameDictionary == {}:
        foundNames = []
        i = 1000
        for name in np.append(visitorSamples[:,0],homeSamples[:,0],0):
            if name not in foundNames:
                foundNames.append(name)
                nameDictionary[name] = i
                i = i + 1
    
    for i in range(0,len(visitorSamples)):
        visitorSamples[i][0] = nameDictionary[visitorSamples[i][0]]
        homeSamples[i][0] = nameDictionary[homeSamples[i][0]]

    return visitorSamples,homeSamples,nameDictionary

def preProcessData(visitorDimentions,homeDimentions,removeMedianData,normilizeData,removeTeamNames,nameDictionary):
    visitorScores = np.asarray(visitorDimentions)[:,-1]
    homeScores = np.asarray(homeDimentions)[:,-1]
    if removeTeamNames: 
        visitorDimentions = np.delete(np.asarray(visitorDimentions),np.s_[::len(visitorDimentions[0])-1], 1)
        homeDimentions = np.delete(np.asarray(homeDimentions),np.s_[::len(homeDimentions[0])-1],1)
    else: 
        visitorDimentions = np.delete(np.asarray(visitorDimentions),len(visitorDimentions[0])-1, 1)
        homeDimentions = np.delete(np.asarray(homeDimentions),len(homeDimentions[0])-1,1)
        visitorDimentions,homeDimentions,nameDictionary = hashTeamNames(visitorDimentions,homeDimentions,nameDictionary)
        

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

    allScores = np.asarray(np.delete(allScores,indices,0),np.float32)
    allSamples = np.delete(allSamples,indices,0)
    if normilizeData:
        allSamples = normilize(allSamples)
        allScores = normilize(allScores)
    return allSamples,allScores,nameDictionary

def readDataset(fileName,nameDictionary):
    lenVisTeamDimentions = 17
    visitorDimentions = []
    homeDimentions = []
    samples = [line.strip('\n').strip(' ').split(',') for line in open(fileName)]
    for j in range(1,len(samples)):
        visitorDimentions.append(samples[j][1:lenVisTeamDimentions+1])
        homeDimentions.append(samples[j][lenVisTeamDimentions+1:])
    removeMedianData = False
    normilizeData = False
    removeTeamNames = False
    return preProcessData(visitorDimentions,homeDimentions,removeMedianData,normilizeData,removeTeamNames,nameDictionary)

def readAllDatasets():
    lenVisTeamDimentions = 17
    visitorDimentions = []
    homeDimentions = []
    nameDictionary = {}
    files = ['2009.csv','2010.csv','2011.csv','2012.csv','2013.csv','2014.csv','2015.csv','2016.csv']
    files = ['../Datasets/'+x for x in files]
    for i in range(0,len(files)):
        samples = [line.strip('\n').strip(' ').split(',') for line in open(files[i])]
        for j in range(1,len(samples)):
            visitorDimentions.append(samples[j][1:lenVisTeamDimentions+1])
            homeDimentions.append(samples[j][lenVisTeamDimentions+1:])
    removeMedianData = False
    normilizeData = False
    removeTeamNames = False
    return preProcessData(homeDimentions,visitorDimentions,removeMedianData,normilizeData,removeTeamNames,nameDictionary)

def divideData(allSamples,allScores):
    trainingSize = int(0.9*len(allSamples))

    trainData = allSamples[:trainingSize]
    testData  = allSamples[trainingSize+1:]

    trainScores = allScores[:trainingSize]
    testScores  = allScores[trainingSize+1:]

    return trainData,testData,trainScores,testScores

