import sys
from Homeutils2 import * 
import tensorflow as tf
import re
import numpy as np
#import matplotlib.pyplot as plt
#tf.InteractiveSession
for x in range(1,len(sys.argv)):
    arg = sys.argv[x]
    vs,hs,features, labels = readFromCSV(arg)

tre=int(np.floor(0.7*len(hs)))
# Initializers
sigma = 1
weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
bias_initializer = tf.zeros_initializer()


data_train=hs[:tre]
data_test=hs[tre:]
xtr = data_train
xte = data_test
ytr=np.asarray(labels[:tre])
yte=np.asarray(labels[tre:])
X=tf.placeholder(dtype=tf.float32)
Y=tf.placeholder(dtype=tf.float32)
n_stocks = len(data_train[0])
n_neurons_1 = 1024
n_neurons_2 = 512
n_neurons_3 = 256
n_neurons_4 = 128
n_target = 1
# Layer 1: Variables for hidden weights and biases
W_hidden_1 = tf.Variable(weight_initializer([n_stocks, n_neurons_1]))
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

#Cost function
mse = tf.reduce_mean(tf.squared_difference(out, Y))

# Optimizer
opt = tf.train.AdamOptimizer().minimize(mse)



# Make Session
net = tf.Session()
# Run initializer
net.run(tf.global_variables_initializer())
# Number of epochs and batch size
epochs = 10
batch_size = 1
y_train = ytr
y_test = yte
X_train = xtr
X_test = xte
for e in range(epochs):

    # Shuffle training data
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    X_train = X_train[shuffle_indices]
    y_train = y_train[shuffle_indices]

    # Minibatch training
    for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = y_train[start:start + batch_size]
        # Run optimizer with batch
        net.run(opt, feed_dict={X: batch_x, Y: batch_y})

        # Show progress
        if np.mod(i, 5) == 0:
            # Prediction
            pred = net.run(out, feed_dict={X: X_test})           
   
predictions = np.round(pred)
accuracy = []
print predictions
print y_test
for i in range(0,len(y_test)): 
    if predictions[0][i] == y_test[i]: 
        accuracy.append(1)
    else: 
        accuracy.append(0)
print accuracy 
print sum(accuracy)/float(len(y_test))
print np.round(pred) 
mse_final = net.run(mse, feed_dict={X: X_test, Y: y_test})
print(mse_final)

