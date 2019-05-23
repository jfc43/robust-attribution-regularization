import numpy as np
import tensorflow as tf
import random
import scipy
import scipy.stats as stats

class SimpleGradientAttack(object):

    def __init__(self, sess, test_image, original_label, NET, attack_method, epsilon,
                 k_top=100, eval_k_top=100, num_steps=100, attack_iters=100, attack_times=3, alpha=1.,  attack_measure="intersection",
                 reference_image=None, target_map = None, same_label = True):
        """
        Args:
            mean_image: The mean image of the data set(The assumption is that the images are mean subtracted)
            sess: Session containing model(and surrogate model's) graphs
            test_image: Mean subtracted test image
            original_label: True label of the image
            NET: Original neural network. It's assumed that NET.saliency is the saliency map tensor and
            NET.saliency_flatten is its flatten version.
            NET2: Surrogate neural network with the same structure and weights of the orignal network but
            with activations replaced by softplus function
            (necessary only when the activation function of the original function
            does not have second order gradients, ex: ReLU). It's assumed that NET.saliency is the
            saliency map tensor and NET2.saliency_flatten is its flatten version.
            k_top: the topK parameter of the attack (refer to the original paper)
            num_steps: Number of steps in Integrated Gradients Algorithm
            reference_image: Mean subtracted reference image of Integrated Gradients Algorithm
        """
        if len(test_image.shape)!=3:
            raise ValueError("Invalid Test Image Dimensions")
        if sum([NET.input.get_shape()[-i]!=test_image.shape[-i] for i in [1,2,3]]):
            raise ValueError("Model's input dimensions is not Compatible with the provided test image!")

        self.status = 1
        self.sess = sess
        self.target_map = target_map
        self.same_label = same_label

        self.NET = NET
        self.w = NET.input.get_shape()[1].value
        self.h = NET.input.get_shape()[2].value
        self.c = NET.input.get_shape()[3].value
        self.num_classes = NET.output_with_relu.get_shape()[-1].value

        self.test_image = test_image
        self.original_label = original_label
        self.num_steps = num_steps

        self.attack_method = attack_method
        self.epsilon = epsilon
        self.attack_iters = attack_iters
        self.attack_times = attack_times
        self.alpha = alpha
        self.attack_measure = attack_measure

        self.k_top = k_top
        self.eval_k_top = eval_k_top

        if self.check_prediction() and same_label:
            self.status = 0
            return

        self.create_attack_ops()

    def check_prediction(self):

        """ If the network's prediction is incorrect in the first place, attacking has no meaning."""
        predicted_scores = self.sess.run(self.NET.output_with_relu, feed_dict={self.NET.input: self.test_image if len(self.test_image.shape)==4 else [self.test_image]})
        self.original_confidence = np.max(predicted_scores)
        if np.argmax(predicted_scores,1) != self.original_label:
            print("Network's Prediction is Already Incorrect!")
            return True
        else:
            return False

    def create_attack_ops(self):

        top_val, self.NET.top_idx = tf.nn.top_k(self.NET.saliency_flatten, self.k_top)
        top_val2, self.NET.top_idx2 = tf.nn.top_k(self.NET.saliency_flatten2, self.k_top)

        self.saliency1, self.topK = self.run_model(self.sess,[self.NET.saliency, self.NET.top_idx], self.test_image, self.NET)
        self.saliency1_flatten = np.reshape(self.saliency1, [self.w * self.h])

        elem1 = np.argsort(np.reshape(self.saliency1, [self.w * self.h]))[-self.k_top:]
        self.elements1 = np.zeros(self.w * self.h)
        self.elements1[elem1] = 1
        topK_loss = tf.reduce_sum((self.NET.saliency_flatten2 * self.elements1))
        self.NET.topK_direction = -tf.gradients(topK_loss, self.NET.input)[0]

        if self.attack_method == 'mass_center' or self.attack_measure == 'mass_center':
            y_mesh, x_mesh = np.meshgrid(np.arange(self.h), np.arange(self.w))

            self.NET.mass_center = tf.stack([tf.reduce_sum(self.NET.saliency * x_mesh)/(self.w * self.h), tf.reduce_sum(self.NET.saliency * y_mesh)/(self.w * self.h)])

            self.NET.mass_center2 = tf.stack([tf.reduce_sum(self.NET.saliency2 * x_mesh)/(self.w * self.h), tf.reduce_sum(self.NET.saliency2 * y_mesh)/(self.w * self.h)])

            self.mass_center1 = self.run_model(self.sess, self.NET.mass_center2, self.test_image, self.NET).astype(int)
            mass_center_loss = -tf.reduce_sum((self.NET.mass_center2 - self.mass_center1)**2)
            self.NET.mass_center_direction= -tf.gradients(mass_center_loss, self.NET.input)[0]

        if self.target_map is not None:
            target_loss = -tf.reduce_sum((self.NET.saliency2 * self.target_map))
            self.NET.target_direction = -tf.gradients(target_loss, self.NET.input)[0]

    def run_model(self, sess, operation, feed, NET):

        if len(feed.shape) == 3:
            return sess.run(operation, feed_dict={NET.input: [feed], NET.label_ph: self.original_label})
        elif len(feed.shape) == 4:
            return sess.run(operation, feed_dict={NET.input: feed, NET.label_ph: self.original_label})
        else:
            raise RuntimeError("Input image shape invalid!")

    def give_simple_perturbation(self, attack_method, in_image):

        if attack_method == "random":
            perturbation = np.random.normal(size=(self.w, self.h, self.c))
        elif attack_method == "topK":
            perturbation = self.run_model(self.sess, self.NET.topK_direction, in_image, self.NET)
            perturbation = np.reshape(perturbation, [self.w, self.h, self.c])
        elif attack_method == "mass_center":
            perturbation = self.run_model(self.sess, self.NET.mass_center_direction, in_image, self.NET)
            perturbation = np.reshape(perturbation, [self.w, self.h, self.c])
        elif attack_method == "target":
            if self.target_map is None:
                raise ValueError("No target region determined!")
            else:
                perturbation = self.run_model(self.sess, self.NET.target_direction, in_image, self.NET)
                perturbation = np.reshape(perturbation, [self.w, self.h, self.c])

        return np.sign(perturbation)


    def apply_perturb(self, in_image, pert, alpha, bound=1.0):

        out_image = self.test_image + np.clip(in_image + alpha * pert - self.test_image, -bound, bound)
        out_image = np.clip(out_image, 0.0, 1.0)
        return out_image

    def check_measure(self, test_image_pert, measure, same_label = True):

        prob = self.run_model(self.sess, self.NET.output_with_relu, test_image_pert, self.NET)

        if np.argmax(prob,1) != self.original_label and same_label:
            return 1

        if measure=="intersection":
            top2 = self.run_model(self.sess, self.NET.top_idx, test_image_pert, self.NET)
            criterion = float(len(np.intersect1d(self.topK, top2))) / self.k_top
        elif measure=="spearman":
            saliency2_flatten = self.run_model(self.sess, self.NET.saliency_flatten, test_image_pert,
                                               self.NET)
            criterion = scipy.stats.spearmanr(self.saliency1_flatten, saliency2_flatten)[0]
        elif measure=="mass_center":
            center2 = self.run_model(self.sess, self.NET.mass_center, test_image_pert, self.NET).astype(int)
            criterion = -np.linalg.norm(self.mass_center1 - center2)
        elif measure=="kendall":
            saliency2_flatten = self.run_model(self.sess, self.NET.saliency_flatten, test_image_pert,
                                               self.NET)
            criterion = stats.kendalltau(self.saliency1_flatten, saliency2_flatten)[0]
        else:
            raise ValueError("Invalid measure!")

        return criterion

    def iterative_attack(self):
        intersections = []
        spearmans = []
        kendalls = []

        for i in range(self.attack_times):
            intersection, spearman, kendall = self.iterative_attack_once()

            intersections.append(intersection)
            spearmans.append(spearman)
            kendalls.append(kendall)

        return np.array(intersections), np.array(spearmans), np.array(kendalls)

    def iterative_attack_once(self):
        """
        Args:
            attack_method: One of "mass_center", "topK" or "random"
            epsilon: set of allowed maximum $ell_infty$ of perturbations, eg:[2,4]
            iters: number of maximum allowed attack iterations
            alpha: perturbation size in each iteration of the attack
            measure: measure for success of the attack (one of "correlation", "mass_center" or "intersection")
        Returns:
            intersection: The portion of the top K salient pixels in the original picture that are in the
            top K salient pixels of the perturbed image devided
            correlation: The rank correlation between saliency maps of original and perturbed image
            center_dislocation: The L2 distance between saliency map mass centers in original and perturbed images
            confidence: The prediction confidence of the perturbed image
        """

        test_image_pert = self.test_image.copy()
        min_criterion = 1.

        for counter in range(self.attack_iters):

            pert = self.give_simple_perturbation(self.attack_method, test_image_pert)
            test_image_pert = self.apply_perturb(test_image_pert, pert, self.alpha, self.epsilon)
            criterion = self.check_measure(test_image_pert, self.attack_measure, self.same_label)

            if criterion < min_criterion:
                min_criterion = criterion
                self.perturbed_image = test_image_pert.copy()

        if min_criterion==1.:
            self.perturbed_image = self.test_image.copy()
            return 1., 1., 1.

        self.saliency2 = self.run_model(self.sess, [self.NET.saliency], self.perturbed_image, self.NET)

        origin_ig = self.saliency1_flatten
        perturbed_ig = np.reshape(self.saliency2, [self.w * self.h])

        spearman = scipy.stats.spearmanr(origin_ig, perturbed_ig)[0]

        kendall = stats.kendalltau(origin_ig, perturbed_ig)[0]

        origin_ig_topK = np.argsort(origin_ig)[-self.eval_k_top:]
        perturbed_ig_topK = np.argsort(perturbed_ig)[-self.eval_k_top:]
        intersection = float(len(np.intersect1d(origin_ig_topK, perturbed_ig_topK))) / self.eval_k_top

        return intersection, spearman, kendall
