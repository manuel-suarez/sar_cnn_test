#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from glob import glob
import os
import argparse

print(tf.__version__)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='GPU flag. 1 = gpu, 0 = cpu')
parser.add_argument('--test_dir', dest='test_dir', default='/content/test', help='test examples are saved here')
parser.add_argument('--test_data', dest='test_data', default='/content/SAR-CNN-test/test_data', help='data set for testing')
parser.add_argument('--real_sar', dest='real_sar', type=int, default=0, help='real data flag. 1 = real, 0 = simulated')
args = parser.parse_args()

checkpoint_dir = '/content/SAR-CNN-test/checkpoint'
if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)
from model import denoiser

def denoiser_test(denoiser):
    if args.real_sar:
        test_data = args.test_data+"/real"
        print(
            "[*] Start testing on real data. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), test_data, args.test_dir))
    else:
        test_data = args.test_data+"/simulated"
        print(
            "[*] Start testing on images with simulated speckle. Working directory: %s. Collecting data from %s and storing test results in %s" % (
            os.getcwd(), test_data, args.test_dir))
    test_files = glob((test_data+'/*.npy').format('float32'))
    denoiser.test(test_files, ckpt_dir=checkpoint_dir, save_dir=args.test_dir, dataset_dir=test_data)

if __name__ == '__main__':
    if args.use_gpu:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            model = denoiser(sess, args.real_sar)
            denoiser_test(model)
    else:
        with tf.Session() as sess:
            model = denoiser(sess, args.real_sar)
            denoiser_test(model)

