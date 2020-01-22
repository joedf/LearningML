from __future__ import absolute_import, division, print_function, unicode_literals

import time, pickle

# disable use of CUDA gpu, uncomment if you have a compatible gpu and CUDA toolkit set up
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Install TensorFlow

import tensorflow as tf

# Load and prepare the MNIST dataset.
mnist = tf.keras.datasets.mnist
# normalizing values from [0-255] range integers to floats ranging [0-1]
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# Build the tf.keras.Sequential model by stacking layers.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train and evaluate the model
start = time.time()
model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
end = time.time()

print('\nTime elapsed (sec) = '+str(round(end - start,2)))
