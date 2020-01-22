# joedf learning from
# https://adventuresinmachinelearning.com/python-tensorflow-tutorial/

# import tensorflow as tf
# https://stackoverflow.com/a/55573434
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

print('\nTHIS SCRIPT DOESNT WORK. IT IS MEANT FOR V1 of TENSORFLOW. EXITING.\n')
exit()

import tensorflow_datasets
mnist = tensorflow_datasets.load('mnist')


# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])


# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# joedf: we have 2 "tensors", ie. 2 "interfaces" for our 3 layer network,
# such that we connections between out input and hidden layer, so tensor #1 or "interface #1"
# and the connections between our hidden layer and out put layer, so tensor #2.
# 
# the stddev param is for the initalization of the weights/bias and with random values


# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1) # z^(l+1) = ( W^(l) dot* x ) + b^(l)
hidden_out = tf.nn.relu(hidden_out) # activation function for  h(l+1) = f( z^(l+1) )


# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))
# softmax applies the exponential function values and creates a "probability" distribution
# with values between [0, 1] where all components sum up to 1.
# it is also somtimes called the "normalized exponential function".


# clip values to avoid log(0) giving NaN, breaking the traning process
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
# using cross entropy cost / loss function for the optimisation / backpropagation
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
								+ (1 - y) * tf.log(1 - y_clipped), axis=1))


# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# start the session
with tf.Session() as sess:
	# initialise the variables
	sess.run(init_op)
	# total_batch = int(len(mnist.train.labels) / batch_size)
	total_batch = int(len(mnist['train']['labels']) / batch_size)
	for epoch in range(epochs):
		avg_cost = 0
		for i in range(total_batch):
			# batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
			batch_x, batch_y = mnist['train'].next_batch(batch_size=batch_size)
			_, c = sess.run([optimiser, cross_entropy], 
							feed_dict={x: batch_x, y: batch_y})
			avg_cost += c / total_batch
		print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))

