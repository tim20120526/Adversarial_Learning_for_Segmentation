import sys
sys.path.append('../')
import tensorflow as tf
from tensorflow.python.ops import init_ops
from data_process import generator
from revised_model import vgg16
import cv2
import os
class Unet_combine_adver:
    def __init__(self):
        print('Revised U-net Network')
        self.input_image = None
        self.input_label = None
        self.cast_image = None
        self.cast_label = None
        self.lamb = None
        self.result_expand = None
        self.loss, self.loss_mean, self.loss_all, self.train_step = [None] * 4
        self.prediction, self.correct_prediction, self.accuracy = [None] * 3
        self.result_conv = {}
        self.result_relu = {}
        self.result_maxpool = {}
        self.result_from_contract_layer = {}
        self.w = {}
        self.b = {}
        self.reuse = False
        self.variables_d = None
        self.variables_g = None
        self.g_loss = None
        self.d_loss = None
        self.final_outputs = None
        self.unet_train = None
        self.d_opt_op = None

    def init_w(self, shape, name):
        with tf.name_scope('init_w'):
            stddev = tf.sqrt(x=2 / (shape[0] * shape[1] * shape[2]))
            w = tf.Variable(initial_value=tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32), name=name)
            tf.add_to_collection(name='g_losses', value=1e-3 * tf.contrib.layers.l2_regularizer(self.lamb)(w))
            return w

    @staticmethod
    def init_b(shape, name):
        with tf.name_scope('init_w'):
            return tf.Variable(initial_value=tf.random_normal(shape=shape, dtype=tf.float32), name=name)

    @staticmethod
    def copy_and_crop_and_merge(result_from_contract_layer, result_from_upsampling):
        # result_from_contract_layer_shape = tf.shape(result_from_contract_layer)
        # result_from_upsampling_shape = tf.shape(result_from_upsampling)
        # result_from_contract_layer_crop = \
        # 	tf.slice(
        # 		input_=result_from_contract_layer,
        # 		begin=[
        # 			0,
        # 			(result_from_contract_layer_shape[1] - result_from_upsampling_shape[1]) // 2,
        # 			(result_from_contract_layer_shape[2] - result_from_upsampling_shape[2]) // 2,
        # 			0
        # 		],
        # 		size=[
        # 			result_from_upsampling_shape[0],
        # 			result_from_upsampling_shape[1],
        # 			result_from_upsampling_shape[2],
        # 			result_from_upsampling_shape[3]
        # 		]
        # 	)
        result_from_contract_layer_crop = result_from_contract_layer
        return tf.concat(values=[result_from_contract_layer_crop, result_from_upsampling], axis=-1)

    def set_up_unet(self, batch_size,IMAGE_SIZE,OUTPUT_SIZE,CLASS_NUM):
        # input
        with tf.name_scope('input'):
            # learning_rate = tf.train.exponential_decay()
            self.input_image = tf.placeholder(
                dtype=tf.float32, shape=[batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], IMAGE_SIZE[2]],
                name='input_images'
            )
            # not using one-hot coding
            self.input_label = tf.placeholder(
                dtype=tf.int32, shape=[batch_size, OUTPUT_SIZE[0], OUTPUT_SIZE[1]], name='input_labels'
            )
            self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
            self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')
        with tf.name_scope('layer'):
            # layer 1
            with tf.name_scope('layer_1'):
                # conv_1:3X3,channel:1-8
                self.w[1] = self.init_w(shape=[3, 3, IMAGE_SIZE[2], 8], name='w_1')
                self.b[1] = self.init_b(shape=[8], name='b_1')
                print(self.input_image)
                result_conv_1 = tf.nn.conv2d(
                    input=self.input_image, filter=self.w[1],
                    strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
                result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[1], name='add_bias'), name='relu_1')
                print(result_relu_1)
                #            self.result_from_contract_layer[1] = result_relu_1
                # conv_2
                #            self.w[2] = self.init_w(shape=[3, 3, 64, 64], name='w_2')
                #            self.b[2] = self.init_b(shape=[64], name='b_2')
                #            result_conv_2 = tf.nn.conv2d(
                #				input=result_relu_1, filter=self.w[2],
                #				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
                #            result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[2], name='add_bias'), name='relu_2')
                self.result_from_contract_layer[1] = result_relu_1  # 该层结果临时保存, 供上采样使用

                # maxpool
                result_maxpool = tf.nn.max_pool(
                    value=result_relu_1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

                # dropout
                result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)
                print(result_dropout)
                "48,64"
            # layer 2
            with tf.name_scope('layer_2'):
                # conv_1:3X3,8-16
                self.w[3] = self.init_w(shape=[3, 3, 8, 16], name='w_3')
                self.b[3] = self.init_b(shape=[16], name='b_3')
                result_conv_1 = tf.nn.conv2d(
                    input=result_dropout, filter=self.w[3],
                    strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
                result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[3], name='add_bias'), name='relu_1')

                # conv_2
                #            self.w[4] = self.init_w(shape=[3, 3, 128, 128], name='w_4')
                #            self.b[4] = self.init_b(shape=[128], name='b_4')
                #            result_conv_2 = tf.nn.conv2d(
                #				input=result_relu_1, filter=self.w[4],
                #				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
                #            result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[4], name='add_bias'), name='relu_2')
                self.result_from_contract_layer[2] = result_relu_1  # 该层结果临时保存, 供上采样使用

                # maxpool
                result_maxpool = tf.nn.max_pool(
                    value=result_relu_1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

                # dropout
                result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)
                print(result_dropout)
                "24,32"
            # layer 3
            with tf.name_scope('layer_3'):
                # conv_1:3X3,channel:16-32
                self.w[5] = self.init_w(shape=[3, 3, 16, 32], name='w_5')
                self.b[5] = self.init_b(shape=[32], name='b_5')
                result_conv_1 = tf.nn.conv2d(
                    input=result_dropout, filter=self.w[5],
                    strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
                result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[5], name='add_bias'), name='relu_1')

                # conv_2
                #            self.w[6] = self.init_w(shape=[3, 3, 256, 256], name='w_6')
                #            self.b[6] = self.init_b(shape=[256], name='b_6')
                #            result_conv_2 = tf.nn.conv2d(
                #				input=result_relu_1, filter=self.w[6],
                #				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
                #            result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[6], name='add_bias'), name='relu_2')
                self.result_from_contract_layer[3] = result_relu_1  # 该层结果临时保存, 供上采样使用

                # maxpool
                result_maxpool = tf.nn.max_pool(
                    value=result_relu_1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

                # dropout
                result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)
                print(result_dropout)
                "12,16"
            # layer 4
            with tf.name_scope('layer_4'):
                # conv_1:3X3,chanel:32-64
                self.w[7] = self.init_w(shape=[3, 3, 32, 64], name='w_7')
                self.b[7] = self.init_b(shape=[64], name='b_7')
                result_conv_1 = tf.nn.conv2d(
                    input=result_dropout, filter=self.w[7],
                    strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
                result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[7], name='add_bias'), name='relu_1')

                # conv_2
                #            self.w[8] = self.init_w(shape=[3, 3, 512, 512], name='w_8')
                #            self.b[8] = self.init_b(shape=[512], name='b_8')
                #            result_conv_2 = tf.nn.conv2d(
                #				input=result_relu_1, filter=self.w[8],
                #				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
                #            result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[8], name='add_bias'), name='relu_2')
                self.result_from_contract_layer[4] = result_relu_1  # 该层结果临时保存, 供上采样使用

                # maxpool
                result_maxpool = tf.nn.max_pool(
                    value=result_relu_1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

                # dropout
                result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)
                print(result_dropout)
                "6,8"
            # layer 5 (bottom)
            with tf.name_scope('layer_5'):
                # conv_1:3X3,channel:64-128
                self.w[9] = self.init_w(shape=[3, 3, 64, 128], name='w_9')
                self.b[9] = self.init_b(shape=[128], name='b_9')
                result_conv_1 = tf.nn.conv2d(
                    input=result_dropout, filter=self.w[9],
                    strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
                result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[9], name='add_bias'), name='relu_1')
                # conv_2
                #            self.w[10] = self.init_w(shape=[3, 3, 1024, 1024], name='w_10')
                #            self.b[10] = self.init_b(shape=[1024], name='b_10')
                #            result_conv_2 = tf.nn.conv2d(
                #				input=result_relu_1, filter=self.w[10],
                #				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
                #            result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[10], name='add_bias'), name='relu_2')
                "6,8"

                # up sample(反卷积)，上采样2倍数
                self.w[11] = self.init_w(shape=[2, 2, 64, 128], name='w_11')
                self.b[11] = self.init_b(shape=[64], name='b_11')
                result_up = tf.nn.conv2d_transpose(
                    value=result_relu_1, filter=self.w[11],
                    output_shape=[batch_size, 12, 16, 64],
                    strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
                result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[11], name='add_bias'), name='relu_3')
                "12,16"
                # dropout
                result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)
                print(result_dropout)

            # layer 6
            with tf.name_scope('layer_6'):
                # copy, crop and merge
                result_merge = self.copy_and_crop_and_merge(
                    result_from_contract_layer=self.result_from_contract_layer[4],
                    result_from_upsampling=result_dropout)
                # print(result_merge)
                # conv_1
                self.w[12] = self.init_w(shape=[3, 3, 128, 64], name='w_12')
                self.b[12] = self.init_b(shape=[64], name='b_12')
                result_conv_1 = tf.nn.conv2d(
                    input=result_merge, filter=self.w[12],
                    strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
                result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[12], name='add_bias'), name='relu_1')

                # conv_2
                #            self.w[13] = self.init_w(shape=[3, 3, 512, 512], name='w_10')
                #            self.b[13] = self.init_b(shape=[512], name='b_10')
                #            result_conv_2 = tf.nn.conv2d(
                #				input=result_relu_1, filter=self.w[13],
                #				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
                #            result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[13], name='add_bias'), name='relu_2')
                # print(result_relu_2.shape[1])

                # up sample
                self.w[14] = self.init_w(shape=[2, 2, 32, 64], name='w_11')
                self.b[14] = self.init_b(shape=[32], name='b_11')
                result_up = tf.nn.conv2d_transpose(
                    value=result_relu_1, filter=self.w[14],
                    output_shape=[batch_size, 24, 32, 32],
                    strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
                result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[14], name='add_bias'), name='relu_3')

                # dropout
                result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)
                print(result_dropout)
                "24,32"
            # layer 7
            with tf.name_scope('layer_7'):
                # copy, crop and merge
                result_merge = self.copy_and_crop_and_merge(
                    result_from_contract_layer=self.result_from_contract_layer[3],
                    result_from_upsampling=result_dropout)

                # conv_1
                self.w[15] = self.init_w(shape=[3, 3, 64, 32], name='w_12')
                self.b[15] = self.init_b(shape=[32], name='b_12')
                result_conv_1 = tf.nn.conv2d(
                    input=result_merge, filter=self.w[15],
                    strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
                result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[15], name='add_bias'), name='relu_1')

                # conv_2
                #            self.w[16] = self.init_w(shape=[3, 3, 256, 256], name='w_10')
                #            self.b[16] = self.init_b(shape=[256], name='b_10')
                #            result_conv_2 = tf.nn.conv2d(
                #				input=result_relu_1, filter=self.w[16],
                #				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
                #            result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[16], name='add_bias'), name='relu_2')

                # up sample
                self.w[17] = self.init_w(shape=[2, 2, 16, 32], name='w_11')
                self.b[17] = self.init_b(shape=[16], name='b_11')
                result_up = tf.nn.conv2d_transpose(
                    value=result_relu_1, filter=self.w[17],
                    output_shape=[batch_size, 48, 64, 16],
                    strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
                result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[17], name='add_bias'), name='relu_3')

                # dropout
                result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)
                print(result_dropout)
                "48,64"
            # layer 8
            with tf.name_scope('layer_8'):
                # copy, crop and merge
                result_merge = self.copy_and_crop_and_merge(
                    result_from_contract_layer=self.result_from_contract_layer[2],
                    result_from_upsampling=result_dropout)

                # conv_1
                self.w[18] = self.init_w(shape=[3, 3, 32, 16], name='w_12')
                self.b[18] = self.init_b(shape=[16], name='b_12')
                result_conv_1 = tf.nn.conv2d(
                    input=result_merge, filter=self.w[18],
                    strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
                result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[18], name='add_bias'), name='relu_1')

                # conv_2
                #            self.w[19] = self.init_w(shape=[3, 3, 128, 128], name='w_10')
                #            self.b[19] = self.init_b(shape=[128], name='b_10')
                #            result_conv_2 = tf.nn.conv2d(
                #				input=result_relu_1, filter=self.w[19],
                #				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
                #            result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[19], name='add_bias'), name='relu_2')

                # up sample
                self.w[20] = self.init_w(shape=[2, 2, 8, 16], name='w_11')
                self.b[20] = self.init_b(shape=[8], name='b_11')
                result_up = tf.nn.conv2d_transpose(
                    value=result_relu_1, filter=self.w[20],
                    output_shape=[batch_size, 96, 128, 8],
                    strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
                result_relu_3 = tf.nn.relu(tf.nn.bias_add(result_up, self.b[20], name='add_bias'), name='relu_3')

                # dropout
                result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)
                print(result_dropout)
                "96,128"
            # layer 9
            with tf.name_scope('layer_9'):
                # copy, crop and merge
                result_merge = self.copy_and_crop_and_merge(
                    result_from_contract_layer=self.result_from_contract_layer[1],
                    result_from_upsampling=result_dropout)

                # conv_1
                self.w[21] = self.init_w(shape=[3, 3, 16, 8], name='w_12')
                self.b[21] = self.init_b(shape=[8], name='b_12')
                result_conv_1 = tf.nn.conv2d(
                    input=result_merge, filter=self.w[21],
                    strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
                result_relu_1 = tf.nn.relu(tf.nn.bias_add(result_conv_1, self.b[21], name='add_bias'), name='relu_1')

                # conv_2
                #            self.w[22] = self.init_w(shape=[3, 3, 64, 64], name='w_10')
                #            self.b[22] = self.init_b(shape=[64], name='b_10')
                #            result_conv_2 = tf.nn.conv2d(
                #				input=result_relu_1, filter=self.w[22],
                #				strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
                #            result_relu_2 = tf.nn.relu(tf.nn.bias_add(result_conv_2, self.b[22], name='add_bias'), name='relu_2')

                # convolution to [batch_size, OUTPIT_IMG_WIDE, OUTPUT_IMG_HEIGHT, CLASS_NUM]
                # CLASS_NUM=2
                self.w[23] = self.init_w(shape=[1, 1, 8, CLASS_NUM], name='w_11')
                self.b[23] = self.init_b(shape=[CLASS_NUM], name='b_11')
                result_conv_3 = tf.nn.conv2d(
                    input=result_relu_1, filter=self.w[23],
                    strides=[1, 1, 1, 1], padding='VALID', name='conv_3')
                # self.prediction = tf.nn.sigmoid(x=tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias'), name='sigmoid_1')
                self.prediction = tf.nn.bias_add(result_conv_3, self.b[23], name='add_bias')
                self.final_outputs = tf.argmax(input=self.prediction, axis=3)
            # print(self.prediction)
            # print(self.input_label)
        self.variables_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='layer')

    # softmax loss
    #        with tf.name_scope('softmax_loss'):
    #			# using one-hot
    #			# self.loss = \
    #			# 	tf.nn.softmax_cross_entropy_with_logits(labels=self.cast_label, logits=self.prediction, name='loss')
    #
    #			# not using one-hot
    #            self.loss = \
    #				tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction, name='loss')
    #            self.loss_mean = tf.reduce_mean(self.loss)
    #            tf.add_to_collection(name='loss', value=self.loss_mean)
    #            self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))
    #
    #		# accuracy
    #        with tf.name_scope('accuracy'):
    #			# using one-hot
    #			# self.correct_prediction = tf.equal(tf.argmax(self.prediction, axis=3), tf.argmax(self.cast_label, axis=3))
    #
    #			# not using one-hot
    #            self.correct_prediction = \
    #				tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32), self.input_label)
    #            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
    #            self.accuracy = tf.reduce_mean(self.correct_prediction)
    #
    #		# Gradient Descent
    #        with tf.name_scope('Gradient_Descent'):
    #            self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_all)
    def loss_fuc(self, batch_size):
        with tf.name_scope('softmax_loss'):
            # using one-hot
            # self.loss = \
            # 	tf.nn.softmax_cross_entropy_with_logits(labels=self.cast_label, logits=self.prediction, name='loss')

            # not using one-hot
            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_label, logits=self.prediction,
                                                                       name='loss')
            self.loss_mean = tf.reduce_mean(self.loss)
            tf.add_to_collection(name='g_losses', value=self.loss_mean)
        #            self.loss_all = tf.add_n(inputs=tf.get_collection(key='loss'))

        g_outputs = self.d(self.prediction, self.input_image, name='g')
        t_outputs = self.d(tf.one_hot(indices=self.input_label, depth=2, on_value=1.0, off_value=0.0, axis=-1),
                           self.input_image, name='t')

        # add each losses to collection
        tf.add_to_collection(
            'g_losses',
            1e-3 * tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([batch_size], dtype=tf.int64),
                    logits=g_outputs)))
        tf.add_to_collection(
            'd_losses',
            1e-3 * tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.ones([batch_size], dtype=tf.int64),
                    logits=t_outputs)))
        tf.add_to_collection(
            'd_losses',
            1e-3 * tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.zeros([batch_size], dtype=tf.int64),
                    logits=g_outputs)))

        self.g_loss = tf.add_n(tf.get_collection('g_losses'), name='total_g_loss')
        self.d_loss = tf.add_n(tf.get_collection('d_losses'), name='total_d_loss')

        batch_rate = tf.Variable(0)

        learning_rate = tf.train.exponential_decay(
            1e-4,
            batch_rate,
            12000,
            0.95,
            staircase=True)

        batch_rate1 = tf.Variable(0)

        learning_rate1 = tf.train.exponential_decay(
            1e-4,
            batch_rate1,
            12000,
            0.95,
            staircase=True)

        batch_rate2 = tf.Variable(0)

        learning_rate2 = tf.train.exponential_decay(
            1e-4,
            batch_rate2,
            12000000,
            0.95,
            staircase=True)
        g_opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        d_opt = tf.train.AdamOptimizer(learning_rate=learning_rate1)
        unet_train_opt = tf.train.AdamOptimizer(learning_rate=learning_rate2)
        self.d_opt_op = d_opt.minimize(self.d_loss, var_list=self.variables_d)
        g_opt_op = g_opt.minimize(self.g_loss, var_list=self.variables_g)
        self.unet_train = unet_train_opt.minimize(self.loss_mean, var_list=self.variables_g)

        with tf.control_dependencies([self.d_opt_op, g_opt_op]):
            self.train_step = tf.no_op(name='train')
        # accuracy
        with tf.name_scope('accuracy'):
            # using one-hot
            # self.correct_prediction = tf.equal(tf.argmax(self.prediction, axis=3), tf.argmax(self.cast_label, axis=3))

            # not using one-hot
            # pixel acc
            self.correct_prediction = tf.equal(tf.argmax(input=self.prediction, axis=3, output_type=tf.int32),
                                               self.input_label)
            self.correct_prediction = tf.cast(self.correct_prediction, tf.float32)
            self.accuracy = tf.reduce_mean(self.correct_prediction)

        # Gradient Descent

    #        with tf.name_scope('Gradient_Descent'):
    #            self.train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.loss_all)

    def train(self, x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size,model_dir,tb_dir,epochs):
        gpuConfig = tf.ConfigProto(allow_soft_placement=True)
        gpuConfig.gpu_options.allow_growth = True
        self.loss_fuc(batch_size)
        tf.summary.scalar("loss", self.loss_mean)
        tf.summary.scalar('accuracy', self.accuracy)
        merged_summary = tf.summary.merge_all()

        global_step = tf.Variable(0, name='global_step', trainable=False)
        all_parameters_saver = tf.train.Saver(var_list=tf.global_variables())
        with tf.Session(config=gpuConfig) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            summary_writer = tf.summary.FileWriter(tb_dir, sess.graph)
            tf.summary.FileWriter(model_dir, sess.graph)
            #            max_acc_valid = 0
            #            for epoch in range(30):
            #                for x,y in generator(x_train,y_train,batch_size,x_train.shape[0],True):
            ##                    print("*****************************************************")
            #                    sess.run(
            #    						[self.unet_train],
            #    						feed_dict={
            #    							self.input_image: x, self.input_label: y, self.keep_prob: 0.6,
            #    							self.lamb: 0.004})
            #                sum_acc = 0.0
            #                sum_los = 0.0
            #                for x,y in generator(x_valid,y_valid,batch_size,x_valid.shape[0],False):
            #                        # print(label)
            #                    lo, acc = sess.run(
            #        						[self.loss_mean, self.accuracy],
            #        						feed_dict={
            #        							self.input_image: x, self.input_label: y, self.keep_prob: 1.0,
            #        							self.lamb: 0.004}
            #        					)
            #                    sum_acc += acc
            #                    sum_los += lo
            #                valid_acc = sum_acc/x_valid.shape[0]*batch_size
            #                valid_los = sum_los/x_valid.shape[0]*batch_size
            #                print('epoch %d, valid_loss: %.6f and valid_accuracy: %.6f' % (epoch, valid_los, valid_acc))
            #
            ##                print('-----------------------------------------------------')
            #
            #                if max_acc_valid <= valid_acc:
            #                    max_acc_valid = valid_acc
            #                    global_step.assign(epoch).eval()
            #                    all_parameters_saver.save(sess=sess, save_path="ckpt_dir" + "/model.ckpt",global_step=global_step)
            #
            #                    sum_acc = 0.0
            ##                    sum_los = 0.0
            #                    cnt = 0
            #                    for x,y in generator(x_test,y_test,batch_size,x_test.shape[0],False):
            #                        image, acc = sess.run(
            #        						[self.final_outputs, self.accuracy],
            #        						feed_dict={
            #        							self.input_image: x, self.input_label: y,
            #        							self.keep_prob: 1.0, self.lamb: 0.004}
            #        					)
            #                        sum_acc += acc
            ##                        sum_los += lo
            ##                        cv2.imwrite(os.path.join(PREDICT_SAVED_DIRECTORY, '%d.jpg' % cnt), image[0] * 255)
            #                        cnt += 1
            #                    test_acc = sum_acc/x_test.shape[0]*batch_size
            ##                    test_los = sum_los/x_test.shape[0]
            #                print('epoch %d test_accuracy: %.6f' % (epoch, test_acc))
            #            for epoch in range(30):
            #                for x,y in generator(x_train,y_train,batch_size,x_train.shape[0],True):
            #                    sess.run([self.d_opt_op],
            #    						feed_dict={
            #    							self.input_image: x, self.input_label: y, self.keep_prob: 0.6,
            #    							self.lamb: 0.004})

            max_acc_valid = 0
            for epoch in range(epochs):
                step = 0
                print("*****************************************************")
                for x, y in generator(x_train, y_train, batch_size, x_train.shape[0], True):
                    lo, acc, summary_str, g_loss, d_loss = sess.run(
                        [self.loss_mean, self.accuracy, merged_summary, self.g_loss, self.d_loss],
                        feed_dict={
                            self.input_image: x, self.input_label: y, self.keep_prob: 1.0,
                            self.lamb: 0.004}
                    )
                    #                    summary_writer.add_summary(summary_str, step)
                    # print('num %d, loss: %.6f and accuracy: %.6f' % (epoch, lo, acc))
                    if step % 1000 == 0:
                        print(
                            'epoch %d, step %d, train_loss: %.6f and train_accuracy: %.6f g_loss:%.6f  d_loss:%.6f' % (
                            epoch, step, lo, acc, g_loss, d_loss))
                    sess.run(
                        [self.train_step],
                        feed_dict={
                            self.input_image: x, self.input_label: y, self.keep_prob: 0.6,
                            self.lamb: 0.004}
                    )
                    step += 1
                summary_writer.add_summary(summary_str, epoch)
                #                print("++++++++++++++++++++++++++++++++++++++++++")

                sum_acc = 0.0
                sum_los = 0.0
                for x, y in generator(x_valid, y_valid, batch_size, x_valid.shape[0], False):
                    # print(label)
                    lo, acc = sess.run(
                        [self.loss_mean, self.accuracy],
                        feed_dict={
                            self.input_image: x, self.input_label: y, self.keep_prob: 1.0,
                            self.lamb: 0.004}
                    )
                    sum_acc += acc
                    sum_los += lo
                valid_acc = sum_acc / x_valid.shape[0] * batch_size
                valid_los = sum_los / x_valid.shape[0] * batch_size
                print('epoch %d, valid_loss: %.6f and valid_accuracy: %.6f' % (epoch, valid_los, valid_acc))
                #                print('-----------------------------------------------------')

                if max_acc_valid <= valid_acc:
                    max_acc_valid = valid_acc
                    global_step.assign(epoch).eval()
                    all_parameters_saver.save(sess=sess, save_path="ckpt_dir" + "/model.ckpt", global_step=global_step)

                    sum_acc = 0.0
                    #                   sum_los = 0.0
                    cnt = 0
                    for x, y in generator(x_test, y_test, batch_size, x_test.shape[0], False):
                        image, acc = sess.run(
                            [self.final_outputs, self.accuracy],
                            feed_dict={
                                self.input_image: x, self.input_label: y,
                                self.keep_prob: 1.0, self.lamb: 0.004}
                        )
                        sum_acc += acc
                        #                        sum_los += lo
                        #                        cv2.imwrite(os.path.join(PREDICT_SAVED_DIRECTORY, '%d.jpg' % cnt), image[0] * 255)
                        cnt += 1
                    test_acc = sum_acc / x_test.shape[0] * batch_size
                #                    test_los = sum_los/x_test.shape[0]
                print('epoch %d test_accuracy_of_max_valid_accuracy: %.6f' % (epoch, test_acc))

            print("Done training")

    def d(self, inputs, input_image, name='d'):
        #        inputs = tf.expand_dims(inputs,-1)
        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.concat([inputs, input_image], axis=3)
        #        input_image = tf.concat([input_image,input_image,input_image],axis=3)

        inputs = (inputs + 1) / 2
        input_image = (input_image + 1) / 2

        def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)

        #        outputs = tf.convert_to_tensor(inputs)
        #        input_image = tf.convert_to_tensor(input_image)
        inputs = tf.image.resize_images(inputs, size=[224, 224], method=0, align_corners=False)
        #        input_image = tf.image.resize_images(input_image, size=[224, 224], method=0, align_corners=False)
        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(inputs)  # input the image
        vgg1 = vgg.prob
        #        with tf.name_scope("content_vgg1"):
        #            vgg1.build(input_image)#input the image
        with tf.name_scope('d' + name), tf.variable_scope('d', reuse=self.reuse):
            # convolution x 4
            #            with tf.variable_scope('conv1'):
            #                outputs = tf.layers.conv2d(inputs, 16, [3, 3], strides=(1, 1), padding='SAME',kernel_initializer=init_ops.truncated_normal_initializer())
            ##                outputs = tf.layers.conv2d(outputs, 32, [3, 3], strides=(2, 2), padding='SAME')
            #                outputs = tf.layers.max_pooling2d(outputs,[2,2],[2,2],padding='SAME')
            #                print(outputs)
            #                outputs = leaky_relu(outputs, name='outputs')
            #            with tf.variable_scope('conv2'):
            #                outputs = tf.layers.conv2d(outputs, 16, [3, 3], strides=(1, 1), padding='SAME',kernel_initializer=init_ops.truncated_normal_initializer())
            #                outputs = tf.layers.max_pooling2d(outputs,[2,2],[2,2],padding='SAME')
            #                print(outputs)
            #                outputs = leaky_relu(outputs, name='outputs')
            #            with tf.variable_scope('conv3'):
            #                outputs = tf.layers.conv2d(outputs, 32, [3, 3], strides=(1, 1), padding='SAME',kernel_initializer=init_ops.truncated_normal_initializer())
            #                outputs = tf.layers.max_pooling2d(outputs,[2,2],[2,2],padding='SAME')
            #                print(outputs)
            #                outputs_1 = leaky_relu(outputs, name='outputs')
            ##
            #            with tf.variable_scope('conv11'):
            #                outputs = tf.layers.conv2d(input_image, 16, [3, 3], strides=(1, 1), padding='SAME',kernel_initializer=init_ops.truncated_normal_initializer())
            ##                outputs = tf.layers.conv2d(outputs, 32, [3, 3], strides=(2, 2), padding='SAME')
            #                outputs = tf.layers.max_pooling2d(outputs,[2,2],[2,2],padding='SAME')
            #                print(outputs)
            #                outputs = leaky_relu(outputs, name='outputs')
            #            with tf.variable_scope('conv21'):
            #                outputs = tf.layers.conv2d(outputs, 16, [3, 3], strides=(1, 1), padding='SAME',kernel_initializer=init_ops.truncated_normal_initializer())
            #                outputs = tf.layers.max_pooling2d(outputs,[2,2],[2,2],padding='SAME')
            #                print(outputs)
            #                outputs = leaky_relu(outputs, name='outputs')
            #            with tf.variable_scope('conv31'):
            #                outputs = tf.layers.conv2d(outputs, 32, [3, 3], strides=(1, 1), padding='SAME',kernel_initializer=init_ops.truncated_normal_initializer())
            #                outputs = tf.layers.max_pooling2d(outputs,[2,2],[2,2],padding='SAME')
            #                print(outputs)
            #                outputs = leaky_relu(outputs, name='outputs')
            #            outputs = tf.concat([outputs_1,outputs],axis=3)
            #            with tf.variable_scope('conv4'):
            #                outputs = tf.layers.conv2d(outputs, 64, [3, 3], strides=(1, 1), padding='SAME',kernel_initializer=init_ops.truncated_normal_initializer())
            #                outputs = tf.layers.max_pooling2d(outputs,[2,2],[2,2],padding='SAME')
            #                print(outputs)
            #                outputs = leaky_relu(outputs, name='outputs')
            #            with tf.variable_scope('conv5'):
            #                outputs = tf.layers.conv2d(outputs, 64, [3, 3], strides=(1, 1), padding='SAME',kernel_initializer=init_ops.truncated_normal_initializer())
            #                outputs = tf.layers.max_pooling2d(outputs,[2,2],[2,2],padding='SAME')
            #                print(outputs)
            #                outputs = leaky_relu(outputs, name='outputs')
            #            with tf.variable_scope('conv6'):
            #                outputs = tf.layers.conv2d(outputs, 128, [3, 3], strides=(1, 1), padding='SAME',kernel_initializer=init_ops.truncated_normal_initializer())
            #                outputs = tf.layers.max_pooling2d(outputs,[2,2],[2,2],padding='SAME')
            #                print(outputs)
            #                outputs = leaky_relu(outputs, name='outputs')
            with tf.variable_scope('classify', reuse=self.reuse):
                #                batch_size = outputs.get_shape()[0].value
                #                outputs = tf.reshape(outputs, [batch_size, -1])
                #                outputs = leaky_relu(tf.layers.dense(outputs, 128, name='outputs',kernel_initializer=init_ops.truncated_normal_initializer()), name='outputs')
                # 1000-64
                outputs = leaky_relu(tf.layers.dense(vgg1, 64, name='outputs2',
                                                     kernel_initializer=init_ops.truncated_normal_initializer()),
                                     name='outputs2')
                outputs = tf.layers.dense(outputs, 2, name='outputs1',
                                          kernel_initializer=init_ops.truncated_normal_initializer())
                outputs = tf.nn.softmax(outputs)
        self.reuse = True
        self.variables_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='d')
        print(self.variables_d)
        return outputs

