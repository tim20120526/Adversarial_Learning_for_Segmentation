# -*- coding: utf-8 -*-
# @Time : 2018/3/23 16:39
# @Author : luodaqin

import numpy as np
import cv2
import random
import sys
import glob

sys.path.append('../')

import tensorflow as tf
import argparse
import os
from revised_model import vgg16
from tensorflow.python.ops import init_ops
from data_process import load_data
from data_process import generator
from revised_model.revised_U_Net_test import Revised_Unet_test

IMAGE_SIZE=(INPUT_IMG_WIDE, INPUT_IMG_HEIGHT, INPUT_IMG_CHANNEL) = (96, 128, 1)
OUTPUT_SIZE=(OUTPUT_IMG_WIDE, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_CHANNEL) = (96, 128, 1)
TRAIN_SET_SIZE = 8
VALIDATION_SET_SIZE = 27
TEST_SET_SIZE = 30
PREDICT_SET_SIZE = 30
EPOCH_NUM = 1
TRAIN_BATCH_SIZE = 10
VALIDATION_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
PREDICT_BATCH_SIZE = 1
PREDICT_SAVED_DIRECTORY = 'My_data/pred'
EPS = 10e-5
CLASS_NUM = 2
CHECK_POINT_PATH = None


def main(args=None):
    parser = argparse.ArgumentParser(description='combine adversarial learning for XX Segmentation')
    parser.add_argument('--data_dir', default=str, help='Path to Dataset file')
    parser.add_argument('--model_dir', default=str, help='Path to save model')
    parser.add_argument('--tb_dir', default=str, help='Path to save TensorBoard log')

    parser = parser.parse_args(args)

    x_train,y_train,x_valid,y_valid,x_test,y_test,name_train,name_valid,name_test = load_data()
    #print (x_valid.shape)
    #print(y_test.shape)
    net = Revised_Unet_test()
#    CHECK_POINT_PATH = os.path.join(parser.model_dir, "model.ckpt")
    net.set_up_unet(PREDICT_BATCH_SIZE,IMAGE_SIZE,OUTPUT_SIZE,CLASS_NUM)
    net.pred(x_train,y_train,x_valid,y_valid,x_test,y_test,name_test,PREDICT_BATCH_SIZE,parser.model_dir,PREDICT_SAVED_DIRECTORY)

	# net.set_up_unet(PREDICT_BATCH_SIZE)
	# net.predict()
if __name__ == '__main__':
    main()
