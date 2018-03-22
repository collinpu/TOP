import tensorflow as tf
def defineNet(numLayers,initialLayer,layerDecay,sigma,n_dimentions):
    
    # Input and Output Placeholders
    X=tf.placeholder(dtype=tf.float32)
    Y=tf.placeholder(dtype=tf.float32)

    # Initializers
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Neuron definitions
    n_target = 1
    neuronDictionary = {}
    neuronDictionary['0'] = n_dimentions 
    for i in range(1,int(numLayers+1)):
        neuronDictionary[str(i)] = int(initialLayer*(layerDecay**i))

    # Layer definitions
    weightDictionary = {} 
    biasDictionary = {}
    hiddenDictionary = {}
    hiddenDictionary['0'] = X
    for i in range(0,int(numLayers)):
        weightDictionary[str(i)] = tf.Variable(weight_initializer([neuronDictionary[str(i)], neuronDictionary[str(i+1)]]))
        biasDictionary[str(i)] = tf.Variable(bias_initializer(neuronDictionary[str(i+1)]))
        hiddenDictionary[str(i+1)] = tf.nn.relu(tf.add(tf.matmul(hiddenDictionary[str(i)], weightDictionary[str(i)]), biasDictionary[str(i)]))

    # Output layer: Variables for output weights and biases
    W_out = tf.Variable(weight_initializer([neuronDictionary[str(int(numLayers)-1)], n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))

    # Output layer (must be transposed)
    out = tf.transpose(tf.add(tf.matmul(hiddenDictionary[str(int(numLayers)-1)], W_out), bias_out))

    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y))

    # Return optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)
    
    return opt,out,mse,X,Y

