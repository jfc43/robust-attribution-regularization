# Robust Attribution Regularization
This project is for the paper: Robust Attribution Regularization. Some codes are from [MNIST Challenge](https://github.com/MadryLab/mnist_challenge), [CIFAR10 Challenge](https://github.com/MadryLab/cifar10_challenge), [Deep traffic sign classification](https://github.com/joshwadd/Deep-traffic-sign-classification), [tflearn oxflower17](https://github.com/tflearn/tflearn/blob/master/tflearn/datasets/oxflower17.py) and [Interpretation of Neural Network is Fragile](https://github.com/amiratag/InterpretationFragility). 

## Preliminaries
It is tested under Ubuntu Linux 16.04.1 and Python 3.6 environment, and requries some packages to be installed:
* [Tensorflow](https://www.tensorflow.org/install)
* [scipy](https://github.com/scipy/scipy)
* [sklearn](https://scikit-learn.org/stable/)
* [numpy](http://www.numpy.org/)
* [scikit-image](https://scikit-image.org/docs/dev/install.html)
* [Pillow](https://pillow.readthedocs.io/en/stable/installation.html)
* [opencv-python](https://pypi.org/project/opencv-python/)
* [six](https://pypi.org/project/six/)
* [pandas](https://pandas.pydata.org/pandas-docs/stable/install.html)

## Downloading Datasets
* [MNIST](http://yann.lecun.com/exdb/mnist/): included in Tensorflow. 
* [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset): we provide scripts to download it. 
* [Flower](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/index.html): we provide scripts to download it.

## Overview of the Code
### Running Experiments
* Before doing experiments, first edit config.json file to specify experiment settings. We provide some template config files like IG-NORM-config.json. You may need to run `mkdir models` before training models. 
* For GTSRB and Flower, you can run prepare_data.sh to get dataset. 
* train_nat.py: the script used to train NATURAL models. 
* train_adv.py: the script used to train Madry's models. 
* train_attribution.py: the script used to train our models(IG-NORM or IG-SUM-NORM). 
* eval_pgd_attack.py: the script used to evaluate NA and AA of the model. 
* eval_attribution_attack.py: the script used to evaluate IN and CO of the model. 
* test_ig_attack.ipynb: ipython notebook used to present demo figures. 

### Parameters in `config.json`
Model configuration:
- `model_dir`: contains the path to the directory of the currently trained/evaluated model.

Data configuration:
- `data_path`: contains the path to the directory of dataset. 

Training configuration:
- `tf_random_seed`: the seed for the RNG used to initialize the network weights.
- `numpy_random_seed`: the seed for the RNG used to pass over the dataset in random order.
- `max_num_training_steps`: the number of training steps.
- `num_output_steps`: the number of training steps between printing progress in standard output.
- `num_summary_steps`: the number of training steps between storing tensorboard summaries.
- `num_checkpoint_steps`: the number of training steps between storing model checkpoints.
- `training_batch_size`: the size of the training batch.
- `step_size_schedule`: learning rate schedule array.
- `weight_decay`: weight decay rate. 
- `momentum`: momentum rate.
- `m`: m in the gradient step.
- `continue_train`: whether continue previous training. Should be True or False.
- `lambda`: lambda in IG-NORM or beta in IG-SUM-NORM. 
- `approx_factor`: (m / approx_factor) = (m in the attack step). 
- `training_objective`: 'ar' for IG-NORM and 'adv_ar' for IG-SUM-NORM. 

Evaluation configuration:
- `num_eval_examples`: the number of examples to evaluate the model on.
- `eval_batch_size`: the size of the evaluation batches.

Adversarial examples configuration:
- `epsilon`: the maximum allowed perturbation per pixel.
- `num_steps` or `k`: the number of PGD iterations used by the adversary.
- `step_size` or `a`: the size of the PGD adversary steps.
- `random_start`: specifies whether the adversary will start iterating from the natural example or a random perturbation of it.
- `loss_func`: the loss function used to run pgd on. `xent` corresponds to the standard cross-entropy loss, `cw` corresponds to the loss function of [Carlini and Wagner](https://arxiv.org/abs/1608.04644), `ar_approx` corresponds to the regularization term of our IG-NORM objective, `adv_ar_approx` corresponds to our IG-SUM-NORM objective. 

Integrated gradient configuration:
- `num_IG_steps`: the number of segments for summation appproximation of IG. 

Attribution robustness configuration:
- `attribution_attack_method`: can be `random`, `topK`, `mass_center` and `target`.
- `attribution_attack_measure`: can be `kendall`, `intersection`, `spearman` and `mass_center`.
- `saliency_type`: can be `ig` or `simple_gradient`.
- `k_top`: the k used for topK attack.
- `eval_k_top`: the k used for evaluation metric -- TopK intersection.
- `attribution_attack_step_size`: step size of attribution attack.
- `attribution_attack_steps`: the number of iterations used by the attack. 
- `attribution_attack_times`: the number of iterations to repeat the attack. 
