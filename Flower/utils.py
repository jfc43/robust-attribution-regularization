import numpy as np
import tensorflow as tf
import random
from skimage import feature, transform
import _pickle as pkl
import matplotlib.pyplot as plt
from pylab import rcParams
import scipy
import scipy.stats as stats
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops

import flower_input
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

EPSILON = 1e-12
MIN_INPUT = np.zeros([1, 128, 128, 3]).astype(np.float32)
MAX_INPUT = 255 * np.ones([1,128, 128,3]).astype(np.float32)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def plot(data, xi=None, cmap=None, axis=plt, percentile=100, dilation=3.0, alpha=0.8):

    dx, dy = 0.05, 0.05
    xx = np.arange(0.0, data.shape[1], dx)
    yy = np.arange(0.0, data.shape[0], dy)
    xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    extent = xmin, xmax, ymin, ymax
    
    cmap_xi = plt.get_cmap('Greys_r')
    cmap_xi.set_bad(alpha=0)
    
    overlay = None
    if xi is not None:
        # Compute edges (to overlay to heatmaps later)
        xi_greyscale = xi if len(xi.shape) == 2 else np.mean(xi, axis=-1)
        in_image_upscaled = transform.rescale(xi_greyscale, dilation, mode='constant')
        edges = feature.canny(in_image_upscaled).astype(float)
        edges[edges < 0.5] = np.nan
        edges[:5, :] = np.nan
        edges[-5:, :] = np.nan
        edges[:, :5] = np.nan
        edges[:, -5:] = np.nan
        overlay = edges

    abs_max = np.percentile(np.abs(data), percentile)
    abs_min = abs_max

    if xi is None:
        axis.imshow(data, extent=extent, interpolation='none', cmap=cmap)
    else:
        axis.imshow(data, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
        
    if overlay is not None:
        axis.imshow(overlay, extent=extent, interpolation='none', cmap=cmap_xi, alpha=alpha)
        
    axis.axis('off')

    axis.xticks([])
    axis.yticks([])

    return axis

def dataReader():
    flower = flower_input.FlowerData('data')
    X = flower.eval_data.xs
    y = flower.eval_data.ys
    return X, y.astype(int)
    
def get_session(number=None):
    config_gpu = tf.ConfigProto()
    config_gpu.gpu_options.allow_growth = True
    return tf.Session(config=config_gpu)

def integrated_gradients(
    sess,
    baseline,
    inp, 
    target_label_index,
    model,
    gradient_func='output_input_gradient',
    steps=50):
    """Computes integrated gradients for a given network and prediction label.
    Integrated gradients is a technique for attributing a deep network's
    prediction to its input features. It was introduced by:
    https://arxiv.org/abs/1703.01365
    In addition to the integrated gradients tensor, the method also
    returns some additional debugging information for sanity checking
    the computation. See sanity_check_integrated_gradients for how this
    information is used.

    This method only applies to classification networks, i.e., networks 
    that predict a probability distribution across two or more class labels.

    Access to the specific network is provided to the method via a
    'predictions_and_gradients' function provided as argument to this method.
    The function takes a batch of inputs and a label, and returns the
    predicted probabilities of the label for the provided inputs, along with
    gradients of the prediction with respect to the input. Such a function
    should be easy to create in most deep learning frameworks.

    Args:
    inp: The specific input for which integrated gradients must be computed.
    target_label_index: Index of the target class for which integrated gradients
      must be computed.
    predictions_and_gradients: This is a function that provides access to the
      network's predictions and gradients. It takes the following
      arguments:
      - inputs: A batch of tensors of the same same shape as 'inp'. The first
          dimension is the batch dimension, and rest of the dimensions coincide
          with that of 'inp'.
      - target_label_index: The index of the target class for which gradients
        must be obtained.
      and returns:
      - predictions: Predicted probability distribution across all classes
          for each input. It has shape <batch, num_classes> where 'batch' is the
          number of inputs and num_classes is the number of classes for the model.
      - gradients: Gradients of the prediction for the target class (denoted by
          target_label_index) with respect to the inputs. It has the same shape
          as 'inputs'.
    baseline: [optional] The baseline input used in the integrated
      gradients computation. If None (default), the all zero tensor with
      the same shape as the input (i.e., 0*input) is used as the baseline.
      The provided baseline and input must have the same shape. 
    steps: [optional] Number of intepolation steps between the baseline
      and the input used in the integrated gradients computation. These
      steps along determine the integral approximation error. By default,
      steps is set to 50.
    Returns:
    integrated_gradients: The integrated_gradients of the prediction for the
      provided prediction label to the input. It has the same shape as that of
      the input.

    The following output is meant to provide debug information for sanity
    checking the integrated gradients computation.
    See also: sanity_check_integrated_gradients
    prediction_trend: The predicted probability distribution across all classes
      for the various (scaled) inputs considered in computing integrated gradients.
      It has shape <steps, num_classes> where 'steps' is the number of integrated
      gradient steps and 'num_classes' is the number of target classes for the
      model.
    """  
    if baseline is None:
        baseline = 0*inp
    assert(baseline.shape == inp.shape)

    # Scale input and compute gradients.
    scaled_inputs = [baseline + (float(i + 1)/steps)*(inp-baseline) for i in range(0, steps)]
    scaled_labels = [target_label_index for i in range(0, steps)]

    if gradient_func == 'loss_input_gradient':
        grads = sess.run(model.loss_input_gradient, feed_dict = {model.input: scaled_inputs, model.label: scaled_labels})  #  shapes: <steps+1, inp.shape>
    else:
        grads = sess.run(model.output_input_gradient, feed_dict = {model.input: scaled_inputs, model.label_ph: target_label_index})

    avg_grads = np.average(grads[:-1], axis=0)
    integrated_gradients = (inp-baseline)*avg_grads  # shape: <inp.shape>
    return integrated_gradients