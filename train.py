# -*- coding: utf-8 -*-
# @Time : 2018/3/23 16:39
# @Author : luodaqin

import numpy as np
import cv2
import random
import sys
import glob

sys.path.append('../')

import argparse
from revised_model.revised_U_Net import Unet_combine_adver
from data_process import load_data


TRAIN_SET_SIZE = 8
VALIDATION_SET_SIZE = 27
TEST_SET_SIZE = 30
PREDICT_SET_SIZE = 30
VALIDATION_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
PREDICT_BATCH_SIZE = 1
CHECK_POINT_PATH = None
IMAGE_SIZE=(INPUT_W, INPUT_H, INPUT_C) = (96, 128, 1)
OUTPUT_SIZE=(OUTPUT_W, OUTPUT_H, OUTPUT_C) = (96, 128, 1)
CLASS_NUM = 2

def main(args=None):

    parser = argparse.ArgumentParser(description='combine adversarial learning for ISIC-2017 Segmentation')
    parser.add_argument('--data_dir', default=str, help='Path to Dataset file')
    parser.add_argument('--model_dir', default=str, help='Path to save model')
    parser.add_argument('--tb_dir', default=str, help='Path to save TensorBoard log')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=10000)
    parser.add_argument('--batch_size', help='Number of BS', type=int, default=10)

    parser = parser.parse_args(args)

    x_train,y_train,x_valid,y_valid,x_test,y_test = load_data()
#    print (x_train.shape)
#    print (y_train.shape)
    Net = Unet_combine_adver()
    Net.set_up_unet(parser.batch_size,IMAGE_SIZE,OUTPUT_SIZE,CLASS_NUM)
    Net.train(x_train,y_train,x_valid,y_valid,x_test,y_test,
                                            parser.batch_size,
                                            parser.model_dir,
                                            parser.tb_dir,
                                            parser.epochs)
   # test
	# net.set_up_unet(parser.batch_size,IMAGE_SIZE,OUTPUT_SIZE,CLASS_NUM)
	# net.predict()
if __name__ == '__main__':
    main()

