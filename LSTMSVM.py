import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
tf.flags.DEFINE_string("log_dir", "log_root","default logging directory")
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
log_dir = FLAGS.log_dir
num_steps = 24 # number of truncated backprop steps ('n' in the discussion above)
batch_size = 5

state_size = 6
learning_rate = 0.1
num_layers = 1
svmC = 1

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

df = pd.read_csv('output.csv', header = 0)
df = df.sort_index(axis=0 ,ascending=True)
df = df.iloc[::-1] #reverse
df = df.loc[df['label'] == 1]
#normalize data CAUSE A LOT OF TIME!!!!
for i in range(int(df.shape[0] / 25)):
	df['close'][i*25:(i+1)*25] = (df['close'][i*25:(i+1)*25] - np.min(df['close'][i*25:(i+1)*25]) )/ (np.max(df['close'][i*25:(i+1)*25]) - np.min(df['close'][i*25:(i+1)*25]))
df.head()


def load(saver, sess, logdir):
	ckpt = tf.train.get_checkpoint_state(logdir)
	if ckpt:
		saver.restore(sess, ckpt.model_checkpoint_path)
		
		return int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
	else:
		return False

def getSubList(lst, partition_size, location):
   
	result = []
	for i,a in enumerate(lst):
		
		if not (i%partition_size == location or (i ==0 and location ==0)):
			result.append(a)
		
			
	return result

def gen_batch(df, batch_size, num_steps):
	closeColumn = list(df['close'])
	labelColumn = list(df['label'])
	raw_x = np.float32(getSubList( closeColumn, num_steps+1, num_steps))
	raw_y = np.float32(getSubList(labelColumn, num_steps+1, 0))
	
	data_length = len(raw_x) 
	# partition raw data into batches and stack them vertically in a data matrix
	batch_partition_length = data_length // batch_size
	print("batchLength "+ str(batch_partition_length))
	data_x = np.zeros([batch_size, batch_partition_length], dtype=np.float32)
	data_y = np.zeros([batch_size, batch_partition_length], dtype=np.float32)
	for i in range(batch_size):
		data_x[i] = raw_x[batch_partition_length * i:batch_partition_length * (i + 1)]
		data_y[i] = raw_y[batch_partition_length * i:batch_partition_length * (i + 1)]
	# further divide batch partitions into num_steps for truncated backprop
	epoch_size = batch_partition_length // (num_steps+1)

	for i in range(epoch_size):
		x = data_x[:, i * num_steps:(i + 1) * num_steps]
		y = data_y[:, i * num_steps:(i + 1) * num_steps]
#		 print (len(x[0]))
#		 print('X')
#		 print(x)
#		 print('Y')
#		 print(y)
		yield (x, y)


def gen_epochs(n, num_steps):
	for i in range(n):
		yield gen_batch(df, batch_size, num_steps)

def main():
	x = tf.placeholder(tf.float32, [batch_size, num_steps], name='input_placeholder')
	y = tf.placeholder(tf.float32, [batch_size, num_steps], name='labels_placeholder')
	lstm = tf.contrib.rnn.BasicLSTMCell(state_size, forget_bias=0.0)
	cell = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers,state_is_tuple=True)
	init_state = cell.zero_state(batch_size, tf.float32)
	outputs= []
	hidden_states = [];
	# state = init_state
	state_placeholder = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])
	l = tf.unstack(state_placeholder, axis=0)
	print("l")

	state = tuple([tf.contrib.rnn.LSTMStateTuple(l[idx][0], l[idx][1])for idx in range(num_layers)])

	print(x)
	print(state)
	with tf.variable_scope("RNN"):
		for time_step in range(num_steps):
			if time_step > 0: tf.get_variable_scope().reuse_variables()
			(cell_output, state) = cell(x[:,time_step:time_step+1], state)
		   	print(state)
			print("cell")
			hidden_states.append(np.transpose(state.h)) #shape(state.h) = 5*6
	#hidden_states = 
	hidden_states = tf.reshape(tf.concat(hidden_states, 1), [-1, state_size]) #144*5
	hidden_states = np.transpose(hidden_states) #5*144
	W = tf.Variable(tf.zeros([state_size*num_steps,1]))
	b = tf.Variable(tf.zeros([batch_size]))
	y_raw = tf.matmul(hidden_states,W) + b
	regularization_loss = 0.5*tf.reduce_sum(tf.square(W)) 
    hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE,1]), 1 - y*y_raw));
    svm_loss = regularization_loss + svmC*hinge_loss;
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)
	predicted_class = tf.sign(y_raw);
    correct_prediction = tf.equal(y,predicted_class)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
			# print(cell_output)  # 5*6(state_size =6) for each time step
			# outputs.append(cell_output)
			
	# output = tf.reshape(tf.concat(outputs, 1), [-1, state_size]) #120*6
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
        training_losses = []
        for idx, epoch in enumerate(gen_epochs(num_epochs, num_steps)):
            training_loss = 0
            training_state = np.zeros([num_layers, 2,batch_size, state_size], dtype=np.float32)
            
            if verbose:
                print("\nEPOCH", idx)
            
            for step, (X, Y) in enumerate(epoch):
                tr_losses, training_loss_, training_state, _,accuracy = \
                    sess.run([losses,
                              total_loss,
                              final_state,
                              train_step,
                              accuracy],
                              feed_dict={x:X, y:Y, state_placeholder:training_state})
                training_loss += training_loss_
                
        print("accuracy: ", accuracy);
        
main()