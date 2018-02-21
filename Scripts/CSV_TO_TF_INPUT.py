import sys
import numpy as np
def readFromCSV(inputFile):
    samples = [line.strip('\n').strip(' ').split(',') for line in open(inputFile)]
    lenVisTeamDimentions = 17
    date = []
    visitorSamples = []
    homeSamples = []
    for i in range(0,len(samples)):
        date.append(samples[i][0])
        visitorSamples.append(samples[i][1:lenVisTeamDimentions+1])
        homeSamples.append(samples[i][lenVisTeamDimentions+1:])

    visitorFeatures = map(list, zip(*visitorSamples))
    homeFeatures = map(list, zip(*homeSamples))

    featureDistionary = {}
    featureDistionary[date[0]] = date[1:]
    featureDistionary['Home'+homeFeatures[0][0]] = homeFeatures[0][1:]
    featureDistionary['Visitor'+visitorFeatures[0][0]] = visitorFeatures[0][1:]

    for i in range(1,len(homeFeatures)):
        featureDistionary['Home'+homeFeatures[i][0]] = np.array([int(x) for x in homeFeatures[i][1:]])
        featureDistionary['Visitor'+visitorFeatures[i][0]] = np.array([int(x) for x in visitorFeatures[i][1:]])

    labels = []
    for i in range(0,featureDistionary['Home Score'].size):
        if featureDistionary['Home Score'][i] > featureDistionary['Visitor Score'][i]: 
            labels.append(1)    #Home
        elif featureDistionary['Home Score'][i] < featureDistionary['Visitor Score'][i]:
            labels.append(0)    #Visitor
        else: 
            labels.append(0.5)    #Tie 
    return featureDistionary, np.array(labels) 
