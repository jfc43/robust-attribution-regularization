"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import re
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from model import Model
from pgd_attack import LinfPGDAttack

os.environ["CUDA_VISIBLE_DEVICES"]="0"

config_file_path = 'config.json'

with open(config_file_path) as config_file:
    config = json.load(config_file)

# Setting up training parameters
tf.set_random_seed(config['random_seed'])
np.random.seed(config['random_seed'])

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
training_objective = config['training_objective']
m = config["m"]
lamb = config["lambda"]
approx_factor = config["approx_factor"]
continue_train = config["continue_train"]

batch_size = config['training_batch_size']

# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model('train', m, lamb, approx_factor)

# Setting up the optimizer
if training_objective == 'ar':
    re_term = model.IG_regularization_term
    train_step = tf.train.AdamOptimizer(1e-4).minimize(model.loss_with_IG_regularizer,
                                                   global_step=global_step)
elif training_objective == 'adv_ar':
    re_term = model.IG_regularization_term
    train_step = tf.train.AdamOptimizer(1e-4).minimize(model.adv_loss_with_IG_regularizer,
                                                   global_step=global_step)
else:
    assert False, ('Unknown training objective.')

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
print('model_dir: {}'.format(model_dir))
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=3)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
merged_summaries = tf.summary.merge_all()

shutil.copy(config_file_path, model_dir)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

with tf.Session(config = tf_config) as sess:
  # Initialize the summary writer, global variables, and our time counter.

  # Set up adversary
  attack = LinfPGDAttack(sess, model, config['epsilon'], config['k'], config['a'], config['random_start'], config['loss_func'])
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)

  if continue_train:
    checkpoint = tf.train.latest_checkpoint(model_dir)
    saver.restore(sess, checkpoint)
    curr_step = int(checkpoint.split('-')[1])
    sess.run(global_step.assign(curr_step))
  else:
    curr_step = 0
    sess.run(tf.global_variables_initializer())

  training_time = 0.0

  # Main training loop
  for ii in range(curr_step, max_num_training_steps):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    x_batch = x_batch.reshape((-1,28,28,1))

    # Compute Adversarial Perturbations
    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch)
    end = timer()
    training_time += end - start

    nat_dict = {model.input: x_batch, model.label: y_batch}
    adv_dict = {model.input: x_batch_adv, model.label: y_batch}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc, nat_loss = sess.run([model.accuracy, model.xent], feed_dict=nat_dict)
      adv_acc, adv_loss = sess.run([model.accuracy, model.xent], feed_dict=adv_dict)
      IG_re = sess.run(re_term, feed_dict={model.input: x_batch, model.adv_input: x_batch_adv, model.label: y_batch})
      print('Step {}:    ({})'.format(ii, datetime.now()), flush=True)
      print('    training nat accuracy {:.4}%, loss {:.4}'.format(nat_acc * 100,nat_loss), flush=True)
      print('    training adv accuracy {:.4}%, loss {:.4}'.format(adv_acc * 100,adv_loss), flush=True)
      print('    training IG term {:.4}'.format(IG_re), flush=True)

      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time), flush=True)
        training_time = 0.0
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=adv_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
    sess.run(train_step, feed_dict={model.input: x_batch, model.adv_input: x_batch_adv, model.label: y_batch})
    end = timer()
    training_time += end - start
