import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
from PIL import Image


import numpy as np
import tensorflow as tf
import tflite_runtime.interpreter as tflite

import time

""" 
    Must have tensorflow with tflite package installed
    Modelfile: model_edgetpu.tflite
    Usage: python3 edgetpu_runtime.py
"""


model_path="model_edgetpu.tflite"
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model_edgetpu.tflite")
# interpreter = tflite.Interpreter(model_path)
interpreter = tflite.Interpreter(model_path, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details, output_details)

# Test the model on random input data.
# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# interpreter.set_tensor(input_details[0]['index'], input_data)
runtime_list = []
zero_time = time.perf_counter()
print("edge tpu runtime stats")
for _ in range(1000):
    # print("#%d", _)
    start = time.perf_counter()
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.int8)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    runtime_list.append([inference_time * 1000])
print()

print("average_time %.2fms" % (time.perf_counter() - zero_time) * 1000)
print(runtime_list)
import csv 
with open('edgetpu.csv', 'w') as f: 
      
    # using csv.writer method from CSV package 
    write = csv.writer(f) 
      
    write.writerows(runtime_list)

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)