#!/usr/bin/env python
"""
 Copyright (C) 2018-2020 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

Note:
Need to install Intel OpenVino Toolkit for this to work.
Do NOT forget to source the environment variables before running this
Model files are mnist.bin, mnist.mapping, and mnist.xml

Usage:
python3 ncs2_runtime.py --model mnist.xml --input <PATH to input image> --device MYRIAD 
"""
from __future__ import print_function
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from openvino.inference_engine import IECore
from matplotlib import pyplot as plt

import time

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True,
                      type=str, nargs="+")
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)
    args.add_argument("--labels", help="Optional. Path to a labels mapping file", default=None, type=str)
    args.add_argument("-nt", "--number_top", help="Optional. Number of top results", default=10, type=int)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = ie.read_network(model=model_xml, weights=model_bin)

    assert len(net.input_info.keys()) == 1, "Sample supports only single input topologies"
    assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input blobs")
    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(args.input)

    # Read and pre-process input images
    n, c, h, w = net.input_info[input_blob].input_data.shape
    images = np.ndarray(shape=(n, c, h, w))
    for _ in range(1000):  # change this number to run desired number of times
        for i in range(n):
            image = cv2.imread(args.input[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(image.shape)
            # image = np.random.random_sample((28, 28, 1)) # If you would like random sample but shouldn't matter
            plt.imshow(image)
            plt.show()
            image = np.reshape(image, (28, 28, 1))
            print(image.shape)
            if image.shape[:-1] != (h, w):
                log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
                image = cv2.resize(image, (w, h))
            image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
            images[i] = image
        log.info("Batch size is {}".format(n))

        # Loading model to the plugin
        log.info("Loading model to the plugin")
        exec_net = ie.load_network(network=net, device_name=args.device)

        # Start sync inference
        log.info("Starting inference in synchronous mode")
        zero_time = time.perf_counter()
        runtime_list = []
        for i in range(1):  # change the outside loop to 1 and this loop to 1000 to test performance on the same random sample
            runtime = time.perf_counter()
            res = exec_net.infer(inputs={input_blob: images})
            this_runtime = time.perf_counter() - runtime
            print("%.2fms" % (this_runtime * 1000))
            runtime_list.append([this_runtime * 1000])
            
        """         
            # Uncomment this chunk if you would like to see results in time

            log.info("Processing output blob")
            res = res[out_blob]
            log.info("Top {} results: ".format(args.number_top))
            if args.labels:
                with open(args.labels, 'r') as f:
                    labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
            else:
                labels_map = None
            classid_str = "classid"
            probability_str = "probability"
            for i, probs in enumerate(res):
                probs = np.squeeze(probs)
                top_ind = np.argsort(probs)[-args.number_top:][::-1]
                print("Image {}\n".format(args.input[i]))
                print(classid_str, probability_str)
                print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
                for id in top_ind:
                    det_label = labels_map[id] if labels_map else "{}".format(id)
                    label_length = len(det_label)
                    space_num_before = (len(classid_str) - label_length) // 2
                    space_num_after = len(classid_str) - (space_num_before + label_length) + 2
                    space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
                    print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
                                                ' ' * space_num_after, ' ' * space_num_before_prob,
                                                probs[id]))
                print("\n")
            log.info("This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n")
        """

            # print("average time %.2fms" % (time.perf_counter() - zero_time))
        # Processing output blob

    import csv # write data to csv
    with open('intel.csv', 'w') as f: 
        
        # using csv.writer method from CSV package 
        write = csv.writer(f) 
        
        write.writerows(runtime_list)

    log.info("Processing output blob")
    res = res[out_blob]
    log.info("Top {} results: ".format(args.number_top))
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
    classid_str = "classid"
    probability_str = "probability"
    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-args.number_top:][::-1]
        print("Image {}\n".format(args.input[i]))
        print(classid_str, probability_str)
        print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
        for id in top_ind:
            det_label = labels_map[id] if labels_map else "{}".format(id)
            label_length = len(det_label)
            space_num_before = (len(classid_str) - label_length) // 2
            space_num_after = len(classid_str) - (space_num_before + label_length) + 2
            space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
            print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
                                          ' ' * space_num_after, ' ' * space_num_before_prob,
                                          probs[id]))
        print("\n")
    log.info("This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n")

if __name__ == '__main__':
    sys.exit(main() or 0)
