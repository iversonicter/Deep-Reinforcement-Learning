# Author: wang yongjie
# Email:  yongjie.wang@ntu.edu.sg

from util import *

class DQN(object):

    def __init__(self, learning_rate, action_space, state_space):

        self.lr = learning_rate
        self.action_space = action_space
        self.state_space = state_space 

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)

        self.input_state = tf.placeholder(tf.float32, [None, 80, 80, 1])
        self.q_values = tf.placeholder(tf.float32, [None, self.action_space])

        self.predictions = self.create_network(self.input_state)
        self.loss = tf.losses.mean_squared_error(labels = self.q_values, predictions = self.prediction)
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def create_network(self, input_x):
        conv1 = conv(input_x, 8, 8, 32, 4, 4, "conv1", True, "VALID")
        conv2 = conv(conv1, 4, 4, 64, 2, 2, "conv2", True, "VALID")
        conv3 = conv(conv2, 3, 3, 64, 1, 1, "conv3", True, "VALID")
        flatten = tf.reshape(conv3, shape = [conv3.get_shape().as_list()[0], -1], name = "flatten")
        fc1 = fc(flatten, flatten.get_shape().as_list()[1], 512, "fc1", True)
        fc2 = fc(fc1, 512, self.action_space, "fc2", False)
        return fc2

    def backward(self, state, q_values):
        self.sess.run(self.optim, feed_dict = {self.input_state : state, self.q_values : q_values}) # update the trainable parameters

    def forward(self, state):
        return self.sess.run(self.predictions, feed_dict = {self.input_state : state})

    def save(self):
        saver = tf.train.Saver()
        saver.save(self.save, saved_Name, step))
        


