import sys
from utils import * 
import tensorflow as tf
tf.InteractiveSession
for x in range(1,len(sys.argv)):
    arg = sys.argv[x]
    features, labels = readFromCSV(arg)
# Define a and b as placeholders
    a = tf.placeholder(dtype=tf.int32)
    b = tf.placeholder(dtype=tf.int32)

# Define the addition
    c = tf.add(a, b)

# Initialize the graph
    graph = tf.Session()
# Run the graph
    sumOfScores = graph.run(c, feed_dict={a: features["Home Score"][0], b: features["Visitor Score"][0]})
    print "This is the sum of the first game's scores: "+str(sumOfScores)   

