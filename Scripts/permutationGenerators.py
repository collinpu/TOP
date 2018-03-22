import numpy as np
def generateParameterPermutations(bestParameters,sweepType):
    if sweepType == 'RunParameters':
        bestRunParameters = bestParameters[:2]
        numEpochs = 2
        numBatchSizes = 2

        epochChange = np.random.randint(1,4,1)
        batchChange = np.random.randint(1,4,1)*5

        epochs,batchSize = bestRunParameters
        epochs,batchSize = [int(epochs),int(batchSize)]
        epochPermutations = np.linspace(max(epochs-epochChange,1),epochs+epochChange,numEpochs)
        batchSizePermutations = np.linspace(max(batchSize-batchChange,1),batchSize+batchChange,numBatchSizes)

        permutations = []
        for i in range(0,numEpochs):
            for j in range(0,numBatchSizes):
                    permutations.append([epochPermutations[i],batchSizePermutations[j]])
        return np.asarray(permutations)

    elif sweepType == 'ArchitectureParameters':
        bestArchitectureParameters = bestParameters[2:]
        numLayerPermutations = 2
        numInitialNeuronPermutations = 2
        numNodeDecayPermutations = 2
        numSigmaPermutations = 2
        
        layerChange = np.random.randint(1,3,1)
        neuronChange = np.random.randint(1,5,1)*128
        decayChange = np.random.randint(1,3,1)*0.125
        sigmaChange = np.random.randint(1,2,1)*10
        
        layers,initialNeurons,nodeDecay,sigma = bestArchitectureParameters
        layers,initialNeurons,nodeDecay,sigma = [int(layers),int(initialNeurons),round(nodeDecay,2),sigma]
        layerPermutations = np.linspace(max(layers-layerChange,1),layers+layerChange,numLayerPermutations)
        intialNeuronPermutations = np.linspace(max(initialNeurons-neuronChange,128),initialNeurons+neuronChange,numInitialNeuronPermutations)
        nodeDecayPermutations = np.linspace(max(nodeDecay-decayChange,0.125),min(nodeDecay+decayChange,1.25),numNodeDecayPermutations)
        sigmaPermutations = np.linspace(sigma/sigmaChange,min(10,sigma*sigmaChange,numSigmaPermutations))

        permutations = []
        for i in range(0,numLayerPermutations):
            for j in range(0,numInitialNeuronPermutations):
                for k in range(0,numNodeDecayPermutations):
                    for l in range(0,numSigmaPermutations): 
                        permutations.append([layerPermutations[i],intialNeuronPermutations[j],nodeDecayPermutations[k],sigmaPermutations[l]])
        return permutations
   

