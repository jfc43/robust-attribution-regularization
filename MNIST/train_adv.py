"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
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

# seeding randomness
tf.set_random_seed(config['random_seed'])
np.random.seed(config['random_seed'])

# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
batch_size = config['training_batch_size']
m = config["m"]
lamb = config["lambda"]

# Setting up the data and the model
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train', m = m, lamb = lamb)

# Setting up the optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(model.loss, global_step=global_step)

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
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

# keep the configuration file with the model for reproducibility
shutil.copy(config_file_path, model_dir)

with tf.Session() as sess:

  # Set up adversary
  attack = LinfPGDAttack(sess, model,
                       config['epsilon'],
                       config['k'],
                       config['a'],
                       config['random_start'],
                       config['loss_func'])

  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  for ii in range(max_num_training_steps):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    x_batch = x_batch.reshape((-1,28,28,1))

    # Compute Adversarial Perturbations
    start = timer()
    x_batch_adv = attack.perturb(x_batch, y_batch)
    end = timer()
    training_time += end - start

    nat_dict = {model.input: x_batch,
                model.label: y_batch}

    adv_dict = {model.input: x_batch_adv,
                model.label: y_batch}
    
    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc, nat_loss = sess.run([model.accuracy, model.xent], feed_dict=nat_dict)
      adv_acc, adv_loss = sess.run([model.accuracy,  model.xent], feed_dict=adv_dict)

      print('Step {}:    ({})'.format(ii, datetime.now()), flush = True)
      print('    training nat accuracy {:.4}%, loss {:.4}'.format(nat_acc * 100, nat_loss), flush = True)
      print('    training adv accuracy {:.4}%, loss {:.4}'.format(adv_acc * 100, adv_loss), flush = True)
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time), flush = True)
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
    sess.run(train_step, feed_dict=adv_dict)
    end = timer()
    training_time += end - start
