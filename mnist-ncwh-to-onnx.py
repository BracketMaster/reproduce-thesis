import keras2onnx
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('mnist-ncwh.h5')
onnx_model = keras2onnx.convert_keras(model, model.name)
keras2onnx.save_model(onnx_model, 'mnist-ncwh.onnx')