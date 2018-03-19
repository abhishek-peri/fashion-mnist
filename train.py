#import tensorflow as tf
#import numpy as np
#import pandas as pd
#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument("-l","--lr")
#parser.add_argument("--batch_size")
#parser.add_argument("--init")
#parser.add_argument("--save_dir")
#args = parser.parse_args()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

#with open("train_data", 'r') as fp:
 #    dat
#a = fp.read()
     
#train_data = tf.reshape(traindata,[-1,28,28,1])
#test_data
#val_data

def cnn_model_fn(features, labels, mode):
  	images = tf.reshape(features["x"],[-1,28,28,1])
	conv1 = tf.layers.conv2d(images,filters=16,kernel_size =[5,5],padding = 'same',activation =tf.nn.relu,name='conv1')
	conv2 = tf.layers.conv2d(conv1,filters=32,kernel_size =[5,5],padding = 'same',activation =tf.nn.relu,name='conv2')
	pool1 = tf.layers.max_pooling2d(inputs= conv2,pool_size =[2,2],strides=2,name='pool1')
	conv3 = tf.layers.conv2d(pool1,filters=64,kernel_size =[5,5],padding = 'same',activation =tf.nn.relu,name='conv3')
	conv4 = tf.layers.conv2d(conv3,filters=128,kernel_size =[5,5],padding = 'same',activation =tf.nn.relu,name='conv4')
	pool2 = tf.layers.max_pooling2d(inputs= conv4,pool_size =[2,2],strides=2,name='pool2')
	conv5 = tf.layers.conv2d(pool2,filters=256,kernel_size =[5,5],padding = 'same',activation =tf.nn.relu,name='conv5')
	flat = tf.reshape(conv5, [-1,7*7*256])
	fc1 = tf.layers.dense(flat,units =1024,activation = tf.nn.relu,name='fc1')
	dropout = tf.layers.dropout(inputs=fc1, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	fc2 = tf.layers.dense(inputs = dropout,units=1024,activation = tf.nn.relu,name='fc2') 
	drop = tf.layers.dropout(inputs=fc2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
	logits = tf.layers.dense(inputs = drop,units =10)

	predictions = {"classes" : tf.argmax(input=logits, axis=1),"probabilities": tf.nn.softmax(logits, name ="softmax_tensor")}
	if mode == tf.estimator.ModeKeys.PREDICT:
	  return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
	onehot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32),depth=10)
	loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,logits=logits)
	if mode == tf.estimator.ModeKeys.TRAIN:
	  optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
	  train_op = optimizer.minimize(loss=loss,global_step = tf.train.get_global_step())
	  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
	
	eval_metric_ops = {"accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])}
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
	df1 = pd.read_csv('train.csv',header=0)
	M = df1.as_matrix()
	M =M[:,785]
	N = df1.as_matrix()
	N = N[:,1:785].astype(np.float32)
	train_data = np.asarray(N)
	train_labels = np.asarray(M, dtype=np.int32)
	df2 = pd.read_csv('test.csv',header=0)
	m = df2.as_matrix()
	m =m[:,784]
	n = df2.as_matrix()
	n = n[:,0:784].astype(np.float32)
	eval_data = np.asarray(n)
	eval_labels = np.asarray(m, dtype=np.int32)
	#df3 = pd.read_csv('val.csv',header=0)
	#h = df3.as_matrix()
	#h =h[:,785]
	#l = df1.as_matrix()
	#l = l[:,1:785].astype(float)
	#val_data = np.asarray(l)
	#val_labels = np.asarray(h, dtype=np.int32)

	classifier = tf.estimator.Estimator(model_fn = cnn_model_fn, model_dir = "convnet_model")
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=200)
	train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":train_data},y=train_labels,batch_size=100,num_epochs=None,shuffle=True)
	classifier.train(input_fn=train_input_fn,steps=5000,hooks=[logging_hook])
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":eval_data},y=eval_labels,num_epochs=1,shuffle=False)
	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)
	with open('results','w') as fp:
	    fp.write(eval_results)

if __name__ == "__main__":
   tf.app.run()
