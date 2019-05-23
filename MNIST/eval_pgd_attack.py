from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Global constants
config_file_path = 'config.json'

with open(config_file_path) as config_file:
    config = json.load(config_file)
num_eval_examples = config['num_eval_examples']
eval_batch_size = config['eval_batch_size']

model_dir = config['model_dir']

# Set upd the data, hyperparameters, and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

model = Model()

global_step = tf.contrib.framework.get_or_create_global_step()

# Setting up the Tensorboard and checkpoint outputs
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
eval_dir = os.path.join(model_dir, 'eval_pgd_attack')
if not os.path.exists(eval_dir):
    os.makedirs(eval_dir)

saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(eval_dir)
print('model_dir: {}'.format(model_dir))
checkpoint = tf.train.latest_checkpoint(model_dir)
print('checkpoint: {}'.format(checkpoint))

with tf.Session() as sess:
    # Restore the checkpoint
    attack = LinfPGDAttack(sess, model, config['epsilon'], config['k'], config['a'], config['random_start'], config['loss_func'])
    saver.restore(sess, checkpoint)

    # Iterate over the samples batch-by-batch
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_corr_nat = 0
    total_corr_adv = 0

    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = mnist.test.images[bstart:bend, :]
        x_batch = x_batch.reshape([-1, 28, 28, 1])
        y_batch = mnist.test.labels[bstart:bend]

        dict_nat = {model.input: x_batch,
                      model.label: y_batch}

        x_batch_adv = attack.perturb(x_batch, y_batch)

        dict_adv = {model.input: x_batch_adv,
                      model.label: y_batch}

        cur_corr_nat, cur_xent_nat = sess.run(
                                          [model.num_correct,model.xent],
                                          feed_dict = dict_nat)
        cur_corr_adv, cur_xent_adv = sess.run(
                                          [model.num_correct,model.xent],
                                          feed_dict = dict_adv)

        total_xent_nat += cur_xent_nat
        total_xent_adv += cur_xent_adv
        total_corr_nat += cur_corr_nat
        total_corr_adv += cur_corr_adv

    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    summary = tf.Summary(value=[
          tf.Summary.Value(tag='xent adv eval', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent adv', simple_value= avg_xent_adv),
          tf.Summary.Value(tag='xent nat', simple_value= avg_xent_nat),
          tf.Summary.Value(tag='accuracy adv eval', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy adv', simple_value= acc_adv),
          tf.Summary.Value(tag='accuracy nat', simple_value= acc_nat)])
    summary_writer.add_summary(summary, global_step.eval(sess))

    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('adversarial: {:.2f}%'.format(100 * acc_adv))
