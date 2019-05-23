# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import dtypes

def per_image_standardization(image):
    """Linearly scales `image` to have zero mean and unit variance.
    This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
    of all values in image, and
    `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.
    `stddev` is the standard deviation of all values in `image`. It is capped
    away from zero to protect against division by 0 when handling uniform images.
    Args:
    image: An n-D Tensor where the last 3 dimensions are
           `[height, width, channels]`.
    Returns:
    The standardized image with same shape as `image`.
    Raises:
    ValueError: if the shape of 'image' is incompatible with this function.
    """
    with ops.name_scope(None, 'per_image_standardization', [image]) as scope:
        image = ops.convert_to_tensor(image, name='image')
        num_pixels = math_ops.reduce_prod(array_ops.shape(image)[1:4])
        image = math_ops.cast(image, dtype=dtypes.float32)
        image_mean = math_ops.reduce_mean(image, axis=[-1, -2, -3], keepdims=True)
        variance = (
            math_ops.reduce_mean(
                math_ops.square(image), axis=[-1, -2, -3], keepdims=True) -
            math_ops.square(image_mean))
        variance = gen_nn_ops.relu(variance)
        stddev = math_ops.sqrt(variance)

        # Apply a minimum normalization that protects us against uniform images.
        min_stddev = math_ops.rsqrt(math_ops.cast(num_pixels, dtypes.float32))
        pixel_value_scale = math_ops.maximum(stddev, min_stddev)
        pixel_value_offset = image_mean

        image = math_ops.subtract(image, pixel_value_offset)
        image = math_ops.div(image, pixel_value_scale, name=scope)
        return image

class Model(object):
    """ResNet model."""

    def __init__(self, mode = 'eval', m = None, lamb = None, approx_factor = 10, create_saliency_op = None):
        """ResNet constructor.

        Args:
          mode: One of 'train' and 'eval'.
        """
        self.mode = mode
        self._build_model(m, lamb, approx_factor, create_saliency_op)

    def add_internal_summaries(self):
        pass

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self, m, lamb, approx_factor, create_saliency_op):
        assert self.mode == 'train' or  self.mode == 'eval'
        """Build the core model within the graph."""
        with tf.variable_scope('input'):
            self.input = tf.placeholder(
                tf.float32,
                shape=[None, 128, 128, 3])
            self.adv_input = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
            self.label = tf.placeholder(tf.int64, shape=[None])
            self.label_ph = tf.placeholder(tf.int32,shape=())

        self.output_with_relu = self.forward(self.input, activation='relu')
        self.output_with_softplus = self.forward(self.input, activation='softplus')

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.label, logits=self.output_with_relu)
        self.xent = tf.reduce_sum(self.loss)
        self.mean_xent = tf.reduce_mean(self.loss)
        self.prediction = tf.argmax(self.output_with_relu, 1)
        self.loss_input_gradient = tf.gradients(self.loss,self.input)[0]
        self.output_input_gradient = tf.gradients(self.output_with_relu[:,self.label_ph], self.input)[0]
        self.y_pred = tf.argmax(self.output_with_relu, 1)
        correct_prediction = tf.equal(self.y_pred, self.label)
        self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.weight_decay_loss = self._decay()

        if create_saliency_op == 'ig':
            self.create_ig_saliency_ops()
        elif create_saliency_op == 'simple_gradient':
            self.create_simple_gradient_saliency_ops()
            
        if self.mode == 'train':
            # attribution regularizer
            self.IG = 0
            self.IG_approx = 0
            for k in range(1, m + 1):
                z = self.input + float(k) / m * (self.adv_input - self.input)
                output = self.forward(z, activation='relu')
                softmax_output = tf.nn.softmax(output)
                loss = tf.reduce_sum(-tf.multiply(tf.one_hot(self.label, 17), tf.log(tf.clip_by_value(softmax_output, 1e-10, 1.0))), axis = 1)
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

    def create_ig_saliency_ops(self):

        self.reference_image = tf.placeholder(tf.float32,shape=(128, 128, 3))

        sum_logits = tf.reduce_sum(self.output_with_relu * tf.expand_dims(tf.one_hot(self.label_ph, 17),0))
        parallel_gradients = tf.gradients(sum_logits, self.input)[0]
        average_gradients = tf.reduce_mean(parallel_gradients, 0)
        difference_multiplied = average_gradients * (self.input[-1] - self.reference_image)
        saliency_unnormalized = tf.reduce_sum(tf.abs(difference_multiplied), -1)
        self.saliency = 128 * 128 * tf.divide(saliency_unnormalized, tf.reduce_sum(saliency_unnormalized))
        # we multiply the normalized salinecy map with image size to make saliency scores of
        #images of different sizes comparable
        self.saliency_flatten = tf.reshape(self.saliency, [128 * 128])

        sum_logits2 = tf.reduce_sum(self.output_with_softplus * tf.expand_dims(tf.one_hot(self.label_ph, 17),0))
        parallel_gradients2 = tf.gradients(sum_logits2, self.input)[0]
        average_gradients2 = tf.reduce_mean(parallel_gradients2, 0)
        difference_multiplied2 = average_gradients2 * (self.input[-1] - self.reference_image)
        saliency_unnormalized2 = tf.reduce_sum(tf.abs(difference_multiplied2), -1)
        self.saliency2 = 128 * 128 * tf.divide(saliency_unnormalized2, tf.reduce_sum(saliency_unnormalized2))
        # we multiply the normalized salinecy map with image size to make saliency scores of
        #images of different sizes comparable
        self.saliency_flatten2 = tf.reshape(self.saliency2,[128 * 128])
        
    def create_simple_gradient_saliency_ops(self):
        
        gradient = tf.gradients(tf.reduce_sum(self.output_with_relu * tf.one_hot(self.label_ph, 17)), self.input)[0]
        saliency_unnormalized = tf.reduce_sum(tf.abs(gradient), -1)
        self.saliency = 128 * 128 * tf.divide(saliency_unnormalized,tf.reduce_sum(saliency_unnormalized))
        # we multiply the normalized salinecy map with image size to make saliency scores of
        #images of different sizes comparable 
        self.saliency_flatten = tf.reshape(self.saliency, [128 * 128])
        
        gradient2 = tf.gradients(tf.reduce_sum(self.output_with_softplus * tf.one_hot(self.label_ph, 17)), self.input)[0]
        saliency_unnormalized2 = tf.reduce_sum(tf.abs(gradient2), -1)
        self.saliency2 = 128 * 128 * tf.divide(saliency_unnormalized2, tf.reduce_sum(saliency_unnormalized2))
        # we multiply the normalized salinecy map with image size to make saliency scores of
        #images of different sizes comparable 
        self.saliency_flatten2 = tf.reshape(self.saliency2, [128 * 128])

    def forward(self, x_input, activation = 'relu'):
        with tf.variable_scope('input', reuse=tf.AUTO_REUSE):
            input_standardized = per_image_standardization(x_input)
            x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]

        res_func = self._residual

        # Uncomment the following codes to use w28-10 wide residual network.
        # It is more memory efficient than very deep residual network and has
        # comparably good performance.
        # https://arxiv.org/pdf/1605.07146v1.pdf
        filters = [16, 16, 32, 64]

        # Update hps.num_residual_units to 9
        with tf.variable_scope('unit_1_0', reuse=tf.AUTO_REUSE):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]), activation,
                       activate_before_residual[0])

        with tf.variable_scope('unit_2_0', reuse=tf.AUTO_REUSE):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]), activation,
                       activate_before_residual[1])

        with tf.variable_scope('unit_3_0', reuse=tf.AUTO_REUSE):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]), activation,
                       activate_before_residual[2])

        with tf.variable_scope('unit_last', reuse=tf.AUTO_REUSE):
            if activation == 'relu':
                x = self._relu(x, 0.1)
            else:
                x = self._softplus(x, 0.1)

            x = self._global_avg_pool(x)

        with tf.variable_scope('logit', reuse=tf.AUTO_REUSE):
            pre_softmax = self._fully_connected(x, 17)

        return pre_softmax

    def _residual(self, x, in_filter, out_filter, stride, activation,
                activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activation == 'relu':
            activation_func = self._relu
        else:
            activation_func = self._softplus

        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = activation_func(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = activation_func(x, 0.1)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = activation_func(x, 0.1)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                  'DW', [filter_size, filter_size, in_filters, out_filters],
                  tf.float32, initializer=tf.random_normal_initializer(
                      stddev=np.sqrt(2.0/n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _softplus(self, x, leakiness=0.0):
        return leakiness * x + (1 - leakiness) * tf.nn.softplus(x)

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
