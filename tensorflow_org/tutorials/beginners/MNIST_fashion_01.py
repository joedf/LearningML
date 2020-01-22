from __future__ import absolute_import, division, print_function, unicode_literals

# modified from https://www.tensorflow.org/tutorials/keras/classification
######################################################################


# disable use of CUDA gpu, uncomment if you have a compatible gpu and CUDA toolkit set up
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# STEPS_PER_EPOCH = 6
# lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
#   0.001,
#   decay_steps=STEPS_PER_EPOCH*1000,
#   decay_rate=1,
#   staircase=False)
# model.compile(optimizer=tf.keras.optimizers.Adam(lr_schedule),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])



model.fit(train_images, train_labels, epochs=5)


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
