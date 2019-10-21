"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Model(object):
    def __init__(self, mode = 'eval', m = None, lamb = None, approx_factor = 10, create_saliency_op = None):
        self.input = tf.placeholder(tf.float32, shape = [None, 28, 28, 1])
        self.adv_input = tf.placeholder(tf.float32, shape = [None, 28, 28 ,1])
        self.label = tf.placeholder(tf.int64, shape = [None])
        self.label_ph = tf.placeholder(tf.int32,shape=())

        # first convolutional layer
        self.W_conv1 = self._weight_variable([5,5,1,32])
        self.b_conv1 = self._bias_variable([32])

        # second convolutional layer
        self.W_conv2 = self._weight_variable([5,5,32,64])
        self.b_conv2 = self._bias_variable([64])

        # first fully connected layer
        self.W_fc1 = self._weight_variable([7 * 7 * 64, 1024])
        self.b_fc1 = self._bias_variable([1024])

        # output layer
        self.W_fc2 = self._weight_variable([1024,10])
        self.b_fc2 = self._bias_variable([10])

        self.output_with_relu = self.forward(self.input, activation='relu')
        self.output_with_softplus = self.forward(self.input, activation='softplus')

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.label, logits=self.output_with_relu)
        self.xent = tf.reduce_sum(self.loss)
        self.loss_input_gradient = tf.gradients(self.loss,self.input)[0]
        self.output_input_gradient = tf.gradients(self.output_with_relu[:,self.label_ph], self.input)[0]
        self.y_pred = tf.argmax(self.output_with_relu, 1)
        self.correct_prediction = tf.equal(self.y_pred, self.label)
        self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        if create_saliency_op == 'ig':
            self.create_ig_saliency_ops()
        elif create_saliency_op == 'simple_gradient':
            self.create_simple_gradient_saliency_ops()
            
        if mode == 'train':
            # attribution regularizer
            self.IG = 0
            self.IG_approx = 0
            for k in range(1, m + 1):
                z = self.input + float(k) / m * (self.adv_input - self.input)
                output = self.forward(z, activation='relu')
                softmax_output = tf.nn.softmax(output)
                loss = tf.reduce_sum(-tf.multiply(tf.one_hot(self.label, 10), tf.log(tf.clip_by_value(softmax_output, 1e-10, 1.0))), axis = 1)
                grad = tf.gradients(loss, z)[0]
                self.IG += grad
                if k % approx_factor ==0:
                    self.IG_approx += grad

            self.IG *= (self.adv_input - self.input) / float(m)
            self.IG_approx *= (self.adv_input - self.input) / float(m / approx_factor)

            self.IG_regularization_term = tf.norm(tf.reshape(self.IG, shape=[-1]), ord=1)
            self.IG_regularization_term_approx = tf.norm(tf.reshape(self.IG_approx, shape=[-1]), ord=1)

            self.loss_with_IG_regularizer = self.xent + lamb * self.IG_regularization_term

            # simple attribution regularizer
            adv_output = self.forward(self.adv_input, activation='relu')
            adv_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.label, logits=adv_output)
            self.IG_simple_regularization_term = tf.norm(tf.reshape(adv_loss - self.loss, shape=[-1]), ord=1)
            self.loss_with_IG_simple_regularizer = self.xent + lamb * self.IG_simple_regularization_term

            # combine loss
            self.adv_xent = tf.reduce_sum(adv_loss)
            self.adv_loss_with_IG_regularizer = self.adv_xent + lamb * self.IG_regularization_term
            self.adv_loss_with_IG_regularizer_approx = self.adv_xent + lamb * self.IG_regularization_term_approx

    def forward(self, x_input, activation='relu'):
        if activation == 'relu':
            activation_func = tf.nn.relu
        else:
            activation_func = tf.nn.softplus

        h_conv1 = activation_func(self._conv2d(x_input, self.W_conv1) + self.b_conv1)
        h_pool1 = self._max_pool_2x2(h_conv1)

        h_conv2 = activation_func(self._conv2d(h_pool1, self.W_conv2) + self.b_conv2)
        h_pool2 = self._max_pool_2x2(h_conv2)

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = activation_func(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)

        return tf.matmul(h_fc1, self.W_fc2) + self.b_fc2

    def create_ig_saliency_ops(self):

        self.reference_image = tf.placeholder(tf.float32,shape=(28, 28, 1))

        sum_logits = tf.reduce_sum(self.output_with_relu * tf.expand_dims(tf.one_hot(self.label_ph, 10),0))
        parallel_gradients = tf.gradients(sum_logits, self.input)[0]
        average_gradients = tf.reduce_mean(parallel_gradients, 0)
        difference_multiplied = average_gradients * (self.input[-1] - self.reference_image)
        saliency_unnormalized = tf.reduce_sum(tf.abs(difference_multiplied),-1)
        self.saliency = 28 * 28 * tf.divide(saliency_unnormalized, tf.reduce_sum(saliency_unnormalized))
        # we multiply the normalized salinecy map with image size to make saliency scores of
        #images of different sizes comparable
        self.saliency_flatten = tf.reshape(self.saliency, [28 * 28])

        sum_logits2 = tf.reduce_sum(self.output_with_softplus * tf.expand_dims(tf.one_hot(self.label_ph, 10),0))
        parallel_gradients2 = tf.gradients(sum_logits2, self.input)[0]
        average_gradients2 = tf.reduce_mean(parallel_gradients2, 0)
        difference_multiplied2 = average_gradients2 * (self.input[-1] - self.reference_image)
        saliency_unnormalized2 = tf.reduce_sum(tf.abs(difference_multiplied2), -1)
        self.saliency2 = 28 * 28 * tf.divide(saliency_unnormalized2, tf.reduce_sum(saliency_unnormalized2))
        # we multiply the normalized salinecy map with image size to make saliency scores of
        #images of different sizes comparable
        self.saliency_flatten2 = tf.reshape(self.saliency2,[28 * 28])
        
    def create_simple_gradient_saliency_ops(self):
        
        gradient = tf.gradients(tf.reduce_sum(self.output_with_relu * tf.one_hot(self.label_ph, 10)), self.input)[0]
        saliency_unnormalized = tf.reduce_sum(tf.abs(gradient), -1)
        self.saliency = 28 * 28 * tf.divide(saliency_unnormalized,tf.reduce_sum(saliency_unnormalized))
        # we multiply the normalized salinecy map with image size to make saliency scores of
        #images of different sizes comparable 
        self.saliency_flatten = tf.reshape(self.saliency, [28 * 28])
        
        gradient2 = tf.gradients(tf.reduce_sum(self.output_with_softplus * tf.one_hot(self.label_ph, 10)), self.input)[0]
        saliency_unnormalized2 = tf.reduce_sum(tf.abs(gradient2), -1)
        self.saliency2 = 28 * 28 * tf.divide(saliency_unnormalized2, tf.reduce_sum(saliency_unnormalized2))
        # we multiply the normalized salinecy map with image size to make saliency scores of
        #images of different sizes comparable 
        self.saliency_flatten2 = tf.reshape(self.saliency2, [28 * 28])

    @staticmethod
    def _weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    @staticmethod
    def _bias_variable(shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    @staticmethod
    def _conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    @staticmethod
    def _max_pool_2x2( x):
        return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')
