import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras2onnx

# quick aliases
Sequential = keras.Sequential
layers = keras.layers
mnist = keras.datasets.mnist
to_categorical = keras.utils.to_categorical
Model = keras.Model

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 1, 28, 28)
X_test = X_test.reshape(10000, 1, 28, 28)

# build model
model = Sequential()
model.add(layers.Conv2D(4, kernel_size=3, activation='relu', input_shape=(1, 28, 28), use_bias=False, data_format="channels_first"))
model.add(layers.Conv2D(4, kernel_size=3, activation='relu', use_bias=False, data_format="channels_first"))
model.add(layers.Conv2D(4, kernel_size=3, activation='relu', use_bias=False, data_format="channels_first"))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

# train model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, to_categorical(y_train), validation_data=(X_test, to_categorical(y_test)), epochs=1)
model.save("mnist-ncwh.h5")