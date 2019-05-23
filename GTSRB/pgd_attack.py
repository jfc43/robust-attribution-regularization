"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


class LinfPGDAttack:
    def __init__(self, sess, model, epsilon, k, a, random_start, loss_func = 'xent'):
        """Attack parameter initialization. The attack performs k steps of
        size a, while always staying within epsilon from the initial
        point."""
        self.sess = sess
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.a = a
        self.rand = random_start
        self.loss_func = loss_func

        if self.loss_func == 'xent':
            loss = model.xent
        elif self.loss_func == 'cw':
            label_mask = tf.one_hot(model.y_input,
                              43,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
            wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        elif self.loss_func == 'ar':
            loss = model.IG_regularization_term
        elif self.loss_func == 'ar_approx':
            loss = model.IG_regularization_term_approx
        elif self.loss_func == 'adv_ar':
            loss = model.adv_loss_with_IG_regularizer
        elif self.loss_func == 'adv_ar_approx':
            loss = model.adv_loss_with_IG_regularizer_approx
        else:
            assert False, ('Unknown loss function.')

        if self.loss_func == 'ar' or self.loss_func == 'ar_approx' or self.loss_func == 'adv_ar' or self.loss_func == 'adv_ar_approx':
            self.grad = tf.gradients(loss, self.model.adv_input)[0]
        elif self.loss_func == 'xent' or self.loss_func == 'cw':
            self.grad = tf.gradients(loss, self.model.input)[0]
        else:
            assert False, ('Unknown loss function.')

    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
        examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        for i in range(self.k):
            if self.loss_func == 'ar' or self.loss_func == 'ar_approx' or self.loss_func == 'adv_ar' or self.loss_func == 'adv_ar_approx':
                grad = self.sess.run(self.grad, feed_dict={self.model.input: x_nat,
                                                           self.model.adv_input: x,
                                                           self.model.label: y})
            elif self.loss_func == 'xent' or self.loss_func == 'cw':
                grad = self.sess.run(self.grad, feed_dict={self.model.input: x,
                                                           self.model.label: y})
            else:
                assert False, ('Unknown loss function.')

            x += self.a * np.sign(grad)
            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
            x = np.clip(x, 0.0, 255.0) # ensure valid pixel range

        return x
