
# joedf: code forked while reading 
# https://adventuresinmachinelearning.com/convolutional-neural-networks-tutorial-tensorflow/
# applying the architecture to the fashion_MNIST dataset instead

from __future__ import absolute_import, division, print_function, unicode_literals


# disable use of CUDA gpu, uncomment if you have a compatible gpu and CUDA toolkit set up
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

import time, datetime

print("\nTensorFlow v{}".format(tf.__version__))

print("Loading and preparing dataset...")
# get data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# define classes
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# scale data
train_images = train_images / 255.0
test_images = test_images / 255.0


model = None
model_filename = "my_model_CNN_01b.h5"

# check a pre-existing model is there
if os.path.isfile(model_filename):
	print("Pre-trained Model exist. Loading model...")
	# Recreate the exact same model, including its weights and the optimizer
	model = tf.keras.models.load_model(model_filename)
	# Show the model architecture
	model.summary()

	print("Testing model ...")
	loss, acc = model.evaluate(test_images,  test_labels, verbose=2)
	print('Restored model, accuracy: {:5.2f}%  -  loss: {}'.format(100*acc,loss))

else:
	print("Creating new model ...")

	model = keras.Sequential([
		keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
		keras.layers.Conv2D(32, (5, 5), strides=(1, 1), activation='relu', padding="same"),
		keras.layers.MaxPooling2D( (2,2), strides=(2,2) ),
		keras.layers.Conv2D(64, (5, 5), strides=(1, 1), activation='relu'),
		keras.layers.MaxPooling2D( (2,2), strides=(2,2) ),
		keras.layers.Flatten(),
		keras.layers.Dense(7* 7 * 64, activation='relu'),
		keras.layers.Dense(1000, activation='relu'),
		keras.layers.Dropout(0.2),
		keras.layers.Dense(10, activation='softmax')
	])

	model.compile(optimizer='adam',
				loss='sparse_categorical_crossentropy',
				metrics=['accuracy'])
	
	model.summary()

	# attach tensorboard
	# see https://www.tensorflow.org/tensorboard/get_started
	log_dir="logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

	start = time.time()
	model.fit(train_images, train_labels, epochs=10,
				callbacks=[tensorboard_callback])

	test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
	end = time.time()

	print('\nTest accuracy:', test_acc)
	# achieves ~91% compared to the ~87% from LearningML\tensorflow_org\tutorials\beginners\MNIST_fashion_01.py (based on the tensoflow tutorial)


	print('\nTime elapsed (sec) = '+str(round(end - start,2)))

	# Save the entire model to a HDF5 file.
	# The '.h5' extension indicates that the model shuold be saved to HDF5.
	print('Saving model ...')
	model.save(model_filename) 
	print('Done.')

