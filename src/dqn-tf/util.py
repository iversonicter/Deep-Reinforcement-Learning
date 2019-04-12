# Author: wang yongjie
# Email:  yongjie.wang@ntu.edu.sg

import tensorflow as tf
import numpy as np

def preprocess_atari(frame):
    '''
    crop->down sample, rgb2gray->erase background -> set paddle
    following karpathy's method https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
    '''
    frame = frame[35:195] # crop
    frame = frame[::2, ::2, 0] # downsample
    frame[frame == 144] = 0 # erase background type 1
    frame[frame == 109] = 0 # erase background type 2
    frame[frame != 0] = 1 # set paddles, ball to 1
    frame = frame.astype(np.float)
    return frame[:, :, np.newaxis]

    
def conv(x, filter_height, filter_width, num_filter, stride_x, stride_y, name, relu = True,  padding = 'SAME'):
    input_channels = int(x.get_shape()[-1])
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", shape = [filter_height, filter_width, input_channels, num_filter], initializer =  tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", shape = [num_filter], initializer = tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(x, weights, [1, stride_y, stride_x, 1], padding)
    bias = tf.nn.bias_add(conv, biases, name = name)
    if relu:
        return tf.nn.relu(bias)
    return bias

def fc(x, num_in, num_out, name, relu = True):
    with tf.variable_scope(name):
        weights = tf.get_variable("weights", shape = [num_in, num_out], initializer = tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable("biases", shape = [num_out], initializer = tf.constant_initializer(0))
    fc = tf.add(tf.matmul(x, weights), biases, name = name)
    if relu:
        return tf.nn.relu(fc)
    return fc

