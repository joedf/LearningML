import pickle, time

# joedf: learning ML from
# https://adventuresinmachinelearning.com

import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import numpy as np
import numpy.random as npr

def convert_y_to_vect(y):
	y_vect = np.zeros((len(y), 10))
	for i in range(len(y)):
		y_vect[i, y[i]] = 1
	return y_vect

def f(x):
	return 1 / (1 + np.exp(-x))
def f_deriv(x):
	return f(x) * (1 - f(x))

def setup_and_init_weights(nn_structure):
	W = {}
	b = {}
	for l in range(1, len(nn_structure)):
		W[l] = npr.random_sample((nn_structure[l], nn_structure[l-1]))
		b[l] = npr.random_sample((nn_structure[l],))
	return W, b

def init_tri_values(nn_structure):
	tri_W = {}
	tri_b = {}
	for l in range(1, len(nn_structure)):
		tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
		tri_b[l] = np.zeros((nn_structure[l],))
	return tri_W, tri_b

def feed_forward(x, W, b):
	h = {1: x}
	z = {}
	for l in range(1, len(W) + 1):
		# if it is the first layer, then the input into the weights is x, otherwise, 
		# it is the output from the last layer
		if l == 1:
			node_in = x
		else:
			node_in = h[l]
		z[l+1] = W[l].dot(node_in) + b[l] # z^(l+1) = W^(l)*h^(l) + b^(l)  
		h[l+1] = f(z[l+1]) # h^(l) = f(z^(l)) 
	return h, z

def calculate_out_layer_delta(y, h_out, z_out):
	# delta^(nl) = -(y_i - h_i^(nl)) * f'(z_i^(nl))
	return -(y-h_out) * f_deriv(z_out)

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
	# delta^(l) = (transpose(W^(l)) * delta^(l+1)) * f'(z^(l))
	return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)

def train_nn(nn_structure, X, y, iter_num=3000, alpha=0.25, lamb=0.0001):
	W, b = setup_and_init_weights(nn_structure)
	cnt = 0
	m = len(y)
	avg_cost_func = []
	print('Starting gradient descent for {} iterations'.format(iter_num))
	while cnt < iter_num:
		if cnt%10 == 0:
			print('\rIteration {} of {}'.format(cnt, iter_num), end="")
		tri_W, tri_b = init_tri_values(nn_structure)
		avg_cost = 0
		for i in range(len(y)):
			delta = {}
			# perform the feed forward pass and return the stored h and z values, to be used in the
			# gradient descent step
			h, z = feed_forward(X[i, :], W, b)
			# loop from nl-1 to 1 backpropagating the errors
			for l in range(len(nn_structure), 0, -1):
				if l == len(nn_structure):
					delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
					avg_cost += np.linalg.norm((y[i,:]-h[l]))
				else:
					if l > 1:
						delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
					# triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
					tri_W[l] += np.dot(delta[l+1][:,np.newaxis], np.transpose(h[l][:,np.newaxis])) 
					# trib^(l) = trib^(l) + delta^(l+1)
					tri_b[l] += delta[l+1]
		# perform the gradient descent step for the weights in each layer
		for l in range(len(nn_structure) - 1, 0, -1):
			# W[l] += -alpha * (1.0/m * tri_W[l])
			# b[l] += -alpha * (1.0/m * tri_b[l])

			# joedf: adding regularisation from tutorial
			W[l] += -alpha * (1.0/m * tri_W[l] + lamb * W[l])
			b[l] += -alpha * (1.0/m * tri_b[l])
		# complete the average cost calculation
		avg_cost = 1.0/m * avg_cost
		avg_cost_func.append(avg_cost)
		cnt += 1
	print("")
	return W, b, avg_cost_func

# modified from https://adventuresinmachinelearning.com/stochastic-gradient-descent/
def train_nn_SGD(nn_structure, X, y, iter_num=3000, alpha=0.25, lamb=0.000):
	W, b = setup_and_init_weights(nn_structure)
	cnt = 0
	m = len(y)
	avg_cost_func = []
	print('Starting stochastic gradient descent for {} iterations'.format(iter_num))
	while cnt < iter_num:
		if cnt%10 == 0:
			print('\rIteration {} of {}'.format(cnt, iter_num), end="")
		tri_W, tri_b = init_tri_values(nn_structure)
		avg_cost = 0
		for i in range(len(y)):
			delta = {}
			# perform the feed forward pass and return the stored h and z values, 
			# to be used in the gradient descent step
			h, z = feed_forward(X[i, :], W, b)
			# loop from nl-1 to 1 backpropagating the errors
			for l in range(len(nn_structure), 0, -1):
				if l == len(nn_structure):
					delta[l] = calculate_out_layer_delta(y[i,:], h[l], z[l])
					avg_cost += np.linalg.norm((y[i,:]-h[l]))
				else:
					if l > 1:
						delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
					# triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
					tri_W[l] = np.dot(delta[l+1][:,np.newaxis],
									   np.transpose(h[l][:,np.newaxis])) 
					# trib^(l) = trib^(l) + delta^(l+1)
					tri_b[l] = delta[l+1]
			# perform the gradient descent step for the weights in each layer
			for l in range(len(nn_structure) - 1, 0, -1):
				W[l] += -alpha * (tri_W[l] + lamb * W[l])
				b[l] += -alpha * (tri_b[l])
		# complete the average cost calculation
		avg_cost = 1.0/m * avg_cost
		avg_cost_func.append(avg_cost)
		cnt += 1
	print("")
	return W, b, avg_cost_func
###


# modified from https://adventuresinmachinelearning.com/stochastic-gradient-descent/#attachment_194
def get_mini_batches(X, y, batch_size):
    random_idxs = npr.choice(len(y), len(y), replace=False)
    X_shuffled = X[random_idxs,:]
    y_shuffled = y[random_idxs]
    mini_batches = [(X_shuffled[i:i+batch_size,:], y_shuffled[i:i+batch_size]) for
                   i in range(0, len(y), batch_size)]
    return mini_batches

def train_nn_MBGD(nn_structure, X, y, bs=100, iter_num=3000, alpha=0.25, lamb=0.000):
	W, b = setup_and_init_weights(nn_structure)
	cnt = 0
	m = len(y)
	avg_cost_func = []
	print('Starting Mini-batch gradient descent for {} iterations'.format(iter_num))
	while cnt < iter_num:
		if cnt%10 == 0:
			print('\rIteration {} of {}'.format(cnt, iter_num), end="")
		tri_W, tri_b = init_tri_values(nn_structure)
		avg_cost = 0
		mini_batches = get_mini_batches(X, y, bs)
		for mb in mini_batches:
			X_mb = mb[0]
			y_mb = mb[1]
			# pdb.set_trace()
			for i in range(len(y_mb)):
				delta = {}
				# perform the feed forward pass and return the stored h and z values, 
				# to be used in the gradient descent step
				h, z = feed_forward(X_mb[i, :], W, b)
				# loop from nl-1 to 1 backpropagating the errors
				for l in range(len(nn_structure), 0, -1):
					if l == len(nn_structure):
						delta[l] = calculate_out_layer_delta(y_mb[i,:], h[l], z[l])
						avg_cost += np.linalg.norm((y_mb[i,:]-h[l]))
					else:
						if l > 1:
							delta[l] = calculate_hidden_delta(delta[l+1], W[l], z[l])
						# triW^(l) = triW^(l) + delta^(l+1) * transpose(h^(l))
						tri_W[l] += np.dot(delta[l+1][:,np.newaxis], 
										  np.transpose(h[l][:,np.newaxis])) 
						# trib^(l) = trib^(l) + delta^(l+1)
						tri_b[l] += delta[l+1]
			# perform the gradient descent step for the weights in each layer
			for l in range(len(nn_structure) - 1, 0, -1):
				W[l] += -alpha * (1.0/bs * tri_W[l] + lamb * W[l])
				b[l] += -alpha * (1.0/bs * tri_b[l])
		# complete the average cost calculation
		avg_cost = 1.0/m * avg_cost
		avg_cost_func.append(avg_cost)
		cnt += 1
	print("")
	return W, b, avg_cost_func
###

def predict_y(W, b, X, n_layers):
	m = X.shape[0]
	y = np.zeros((m,))
	for i in range(m):
		h, z = feed_forward(X[i, :], W, b)
		y[i] = np.argmax(h[n_layers])
	return y


############################### end defs ##########################################

if __name__ == '__main__':
	# main thread
	
	# load data
	print("loading data...")
	from sklearn.datasets import load_digits
	digits = load_digits()

	print('preparing data...')
	X_scale = StandardScaler()
	X = X_scale.fit_transform(digits.data)
	y = digits.target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

	y_v_train = convert_y_to_vect(y_train)
	y_v_test = convert_y_to_vect(y_test)

	print('Ready.\n')

	# define NN structure
	# Layer 1: n# of input nodes 
	# Layer 2: n# of hidden nodes 
	# Layer 3: n# of output nodes 
	
	# nn_structure = [64, 30, 10]  # original

	# based on tests from the tutorial, we know that 50 hidden layers and ...
	nn_structure = [64, 50, 10]

	# ... a different learning rate and regularisation parameter will work better
	nn_alpha = 0.5
	nn_lamb = 0.001

	# number of iterations for training
	# nn_iterations = 3000
	# SGD converges much more rapidly than standard GD (Batch Gradient Descent), so we dont need as many iterations.
	# However on the long run, BGD can outperform SGD here at 3000+ iterations.
	nn_iterations = 500

	# note: each iteration of SGD seems to be slower than BGD, but converges much faster
	# so a trade-off method between SGD and BGD is to use Mini-batch Gradient Descent (MBGD)
	# tests from the tutorial show that MBGD seems to far superior an accuracy of ~98% in only 150 iterations
	# versus the next best BGD which requires a lot more iterations (~3000) to reach ~96% accuracy.

	# prompt user to Train or predict (if model available)
	flag=1
	while flag:
		p = input("[T]rain or [P]redict?").upper()
		if p[0] == 'T' or p[0] == 'P':
			flag=0

	if p[0] == 'T':
		start = time.time()
		W, b, avg_cost_func = train_nn_MBGD(nn_structure, X_train, y_v_train, iter_num=nn_iterations, alpha=nn_alpha, lamb=nn_lamb)
		end = time.time()
		print('Time elapsed (sec) = '+str(round(end - start,2)))

		# show plot of cost function
		plt.plot(avg_cost_func)
		plt.ylabel('Average J')
		plt.xlabel('Iteration number')
		plt.show()


		export = input('Export model (y/n)?').upper()

		# joedf: export model
		if export[0] == 'Y':
			print("Exporting model files...")
			pickle.dump( W, open( "weights.p", "wb" ) )
			pickle.dump( b, open( "biases.p", "wb" ) )
	else:
		# load model
		print('loading model...')
		W = pickle.load( open( "weights.p", "rb" ) )
		b = pickle.load( open( "biases.p", "rb" ) )
		print('loaded.')

		from sklearn.metrics import accuracy_score
		print('predicting...')
		start = time.time()
		y_pred = predict_y(W, b, X_test, 3)
		end = time.time()
		acc = accuracy_score(y_test, y_pred)*100
		print('accuracy = '+str(acc))
		print('Time elapsed (sec) = '+str(round(end - start,2)))


	print('Done.')
