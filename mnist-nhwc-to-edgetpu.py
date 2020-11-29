import tensorflow as tf
from tensorflow import keras
import numpy as np

# quick aliases
Sequential = keras.Sequential
layers = keras.layers
mnist = keras.datasets.mnist
to_categorical = keras.utils.to_categorical
Model = keras.Model

# load data to train on
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

# build model
model = Sequential()
model.add(layers.Conv2D(4, kernel_size=3, activation='relu', input_shape=(28,28,1), use_bias=False))
model.add(layers.Conv2D(4, kernel_size=3, activation='relu', use_bias=False))
model.add(layers.Conv2D(4, kernel_size=3, activation='relu', use_bias=False))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

# compile and save
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, to_categorical(y_train), validation_data=(X_test, to_categorical(y_test)), epochs=7)
model.save("mnist.h5")

# convert to tf-lite
def representative_data_gen():
    for image in X_test:
        yield [image.reshape(1,28,28,1).astype(np.float32)]

model = keras.models.load_model('mnist.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.target_spec.supported_types = [tf.int8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

# Save the model.
with open('mnist-nhwc.tflite', 'wb') as f:
  f.write(tflite_model)