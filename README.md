# TLDR

If you are just here for the models, you can download the
following models with a git clone of the repository:

 - ``mnist-ncwh.onnx``
 - ``mnist-nhwc_edgetpu.tflite``

Note that the CONV layers in 
``mnist.tflite`` differ slightly from the others in that
they are in the NHWC format while the others are in the \
NCWH format.

# Purpose

This repository contains some code to help reproduce the
results from my master thesis. I needed to test the same
mnist CNN model on the following devices:

 - edge TPU
 - intel compute stick
 - MAERI DNN accelerator I designed for my thesis

An issue arose when attempting to convert the keras mnist
model to the ONNX format which both the intel compute stick
and my thesis accelerator support. Namely, onnx doesn't 
support the NHWC image format that keras defaults to. Onnx gets
around this limitation by inserting a transpose which both 
Intel's OpenVino and my MAERI compiler have trouble reason about.

It is possible to have keras instead do NCWH by passing 
``data_format="channels_first"`` as an argument to the
keras ``layers.Conv2D``, but certain TensorFlow backends 
are unable to compile such a model for training. It just
so happens that Apple's Metal optimized backend can train
this model whilst the standard TF x86 backend cannot.

To make matters worse however, the keras2onnx utility 
works with tensorflow2.3.1 but doesn't
work with models exported by the version of keras in Apple's
Metal optimized tensorflow. As of Nov-28-2020, tensorflow has
no wheel in pypi for MacOS Big Sur, so it seems Apple's
tensorflow is the only working version of tensorflow for BigSur.

The next section provides a somewhat convoluted process for
reproducing the models in this repo.

# Reproducing
The following sub-sections must be done in order.

## Building ``mnist-ncwh.onnx``
0. Change into the directory where you cloned this repository.
1. Install TF on MacOS using this [link](https://github.com/apple/tensorflow_macos)
which essentially has you run this command: ``/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/apple/tensorflow_macos/master/scripts/download_and_install.sh)"``
2. Source the virtual environment where you installed Apple's Tensorflow.
3. Run ``python3 train-mnist-ncwh.py`` to train and save the NCWH mnist model.
4. Run ``docker run -v `pwd`:/compiler -it ubuntu:20.04`` to fire up a ubuntu
20.04 container and link volumes to your current directory.
5. Run the following commands in the ubuntu container:
```bash
cd compiler
apt update
apt install -y python3 python3-pip curl
```
6. And then these commands
```
pip3 install tensorflow==2.3.1 keras2onnx
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
apt-get update
apt-get install -y edgetpu-compiler
```
7. Run ``python3 mnist-ncwh-to-onnx.py`` to build the onnx model.

## Building ``mnist-nhwc_edgetpu.tflite``
8. Run ``python3 mnist-nhwc-to-edgetpu.py`` still inside the container.
9. Finally, run ``edgetpu_compiler mnist-nhwc.tflite`` inside the container.