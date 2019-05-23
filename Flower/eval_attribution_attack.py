from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import numpy as np
import shutil
import json
import math
import os
import sys
import time

import tensorflow as tf
import flower_input
from model import Model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Global constants
with open('config.json') as config_file:
    config = json.load(config_file)

num_eval_examples = config['num_eval_examples']
epsilon = config['epsilon']
random_seed = config['np_random_seed']
model_dir = config['model_dir']
num_IG_steps = config['num_IG_steps']
k_top = config['k_top']
eval_k_top = config['eval_k_top']
saliency_type = config['saliency_type']
attribution_attack_method = config['attribution_attack_method']
attribution_attack_measure = config['attribution_attack_measure']
attribution_attack_step_size = config['attribution_attack_step_size']
attribution_attack_steps = config['attribution_attack_steps']
attribution_attack_times = config['attribution_attack_times']
data_path = config['data_path']

if saliency_type == 'ig':
    from ig_attack import IntegratedGradientsAttack as SaliencyAttack
elif saliency_type == 'simple_gradient':
    from simple_gradient_attack import SimpleGradientAttack as SaliencyAttack
else:
    assert False, ('Unknown saliency type.')

np.random.seed(random_seed)

# Set upd the data, hyperparameters, and the model
flower = flower_input.FlowerData(data_path)

reference_image = np.zeros((128, 128, 3))

model = Model(mode='eval', create_saliency_op=saliency_type)

saver = tf.train.Saver()

global_step = tf.contrib.framework.get_or_create_global_step()

checkpoint = tf.train.latest_checkpoint(model_dir)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

with tf.Session(config = tf_config) as sess:
    # Restore the checkpoint
    saver.restore(sess, checkpoint)

    test_images = flower.eval_data.xs
    test_labels = flower.eval_data.ys
    
    min_intersections = []
    min_spearmans = []
    min_kendalls = []
    
    correct_cnt = 0

    for i in range(num_eval_examples):
        test_image = test_images[i]
        original_label = test_labels[i]

        module = SaliencyAttack(sess = sess, test_image = test_image, original_label = original_label, NET = model,
                                           attack_method = attribution_attack_method, epsilon = epsilon,
                                           k_top = k_top, eval_k_top = eval_k_top, num_steps = num_IG_steps,
                                           attack_iters = attribution_attack_steps,
                                           attack_times = attribution_attack_times,
                                           alpha = attribution_attack_step_size,  attack_measure = attribution_attack_measure,
                                           reference_image = reference_image, same_label = True)

        if module.status == 1:
            
            correct_cnt += 1
            
            intersections, spearmans, kendalls = module.iterative_attack()
            
            idx = np.argmin(kendalls)
            min_intersections.append(intersections[idx])
            min_spearmans.append(spearmans[idx])
            min_kendalls.append(kendalls[idx])
            
            res_str = '{} {} '.format(i, 1)

            for k in range(attribution_attack_times):
                res_str += '{:.6f} {:.6f} {:.6f} '.format(intersections[k], spearmans[k], kendalls[k])
                

            print('progress: {}/{}, {}'.format(i + 1, num_eval_examples, res_str))
        else:
            res_str = '{} {} '.format(i, 0)

            for k in range(attribution_attack_times):
                res_str += '{:.6f} {:.6f} {:.6f} '.format(0, 0, 0)

            print('progress: {}/{}, prediction incorrect!'.format(i + 1, num_eval_examples))
            
avg_intersection = np.mean(min_intersections)
avg_spearman = np.mean(min_spearmans)
avg_kendall = np.mean(min_kendalls)

print('process {} examples'.format(num_eval_examples))
print('accuracy {}'.format(float(correct_cnt)/num_eval_examples))
print('Average top-k intersection: {:.4f}'.format(avg_intersection))
print('Average spearman rank correlation: {:.4f}'.format(avg_spearman))
print('Average kendall rank correlation: {:.4f}'.format(avg_kendall))

