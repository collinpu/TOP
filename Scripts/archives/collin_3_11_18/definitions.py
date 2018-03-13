import tensorflow as tf
def defineNet(sigma):     
    # Input and Output Placeholders
    X=tf.placeholder(dtype=tf.float32)
    Y=tf.placeholder(dtype=tf.float32)

    # Initializers
    weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
    bias_initializer = tf.zeros_initializer()

    # Layer definitions
    n_dimentions = 15
    n_neurons_1 = 1024
    n_neurons_2 = 512
    n_neurons_3 = 256
    n_neurons_4 = 128
    n_target = 1

    # Layer 1: Variables for hidden weights and biases
    W_hidden_1 = tf.Variable(weight_initializer([n_dimentions, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    # Layer 2: Variables for hidden weights and biases
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    # Layer 3: Variables for hidden weights and biases
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    # Layer 4: Variables for hidden weights and biases
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))

    # Output layer: Variables for output weights and biases
    W_out = tf.Variable(weight_initializer([n_neurons_4, n_target]))
    bias_out = tf.Variable(bias_initializer([n_target]))

    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

    # Output layer (must be transposed)
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))

    # Cost function
    mse = tf.reduce_mean(tf.squared_difference(out, Y))

    # Return optimizer
    opt = tf.train.AdamOptimizer().minimize(mse)
    
    return opt,out,mse,X,Y
