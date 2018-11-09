from __future__ import division, print_function, absolute_import

import sys
import queue
import numpy as np
import threading
import time
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support
import random
import tensorflow as tf
import skimage
import skimage.io
#import matplotlib.pyplot as plt
import imageio
import os

class SegCrop(object):
    def __init__(self, h_offset, w_offset, patch_height, patch_width, img, mask):
        self.h_offset = h_offset
        self.w_offset = w_offset
        self.height = patch_height
        self.width = patch_width
        self.img = img
        self.mask = mask

        y2 = int(self.h_offset + self.height)
        x2 = int(self.w_offset + self.width)
        y1 = self.h_offset
        x1 = self.w_offset
        self.cropped_img = self.img[x1:x2,y1:y2]
        self.cropped_msk = self.mask[x1:x2,y1:y2]

        self.pred_labels = None # ready to predict

        self.fullsz_allone = None

    def direct_fillback(self, prd_lbl=None):

        if prd_lbl is None:
            prd_lbl = self.pred_labels

        pred_2d = np.reshape(prd_lbl, [self.height, self.width])
        full_labels = np.zeros(self.mask.shape, dtype=np.float32)

        for i in range(0, self.height):
            for j in range(0, self.width):
                #print(i + self.h_offset, j + self.w_offset, i, j, full_labels[i + self.h_offset][j + self.w_offset], pred_2d[i][j])
                full_labels[i + self.w_offset][j + self.h_offset] = pred_2d[i][j]

        return full_labels

    def get_allone_patch(self):

        if self.fullsz_allone is None:
            full_labels = np.zeros(self.mask.shape, dtype=np.float32)
            for i in range(0, self.height):
                for j in range(0, self.width):
                    full_labels[i + self.w_offset][j + self.h_offset] = 1
            self.fullsz_allone = full_labels

        return self.fullsz_allone

class SegImage(object):
    def __init__(self, data_dir, fname, img_size, img, mask):
        self.data_dir = data_dir
        self.fname = fname
        self.img_size = img_size
        self.img = img
        self.mask = mask

        self.pred_lbl = None

        # about crop
        self.crop_imgs = []
        self.crop_features = None
        self.crop_gt_labels = None

    def get_crop_count(self):
        return len(self.crop_imgs)

    def get_all_crop_imgs(self):
        if self.crop_features is None:
            features = []
            gt_labels = []
            for crop in self.crop_imgs:
                features.append(crop.cropped_img)
                gt_labels.append(crop.cropped_msk.flatten())
            self.crop_features = np.array(features, np.float32)
            self.crop_gt_labels = np.array(gt_labels, np.float32)
        return self.crop_features, self.crop_gt_labels

    def crop_image_random(self, patch_height, patch_width, crop_count):

        self.crop_imgs.clear()

        h = self.img.shape[0]
        w = self.img.shape[1]
        y1Max = int(h - patch_height)
        x1Max = int(w - patch_width)
        possible_crops = []
        for y1 in range(0, y1Max):
            for x1 in range(0, x1Max):
                possible_crops.append((y1, x1))

        np.random.shuffle(possible_crops)
        for i in range(0, crop_count):
            offset_pair = possible_crops[i]
            cropped = SegCrop(possible_crops[i][0], possible_crops[i][1], patch_height, patch_width, self.img, self.mask)
            self.crop_imgs.append(cropped)

        # print('Random cropped ', len(self.crop_imgs), ' image crops.')
        return self.crop_imgs

    def crop_image_full(self, patch_height, patch_width):
        self.crop_imgs.clear()

        h = self.img.shape[0]
        w = self.img.shape[1]
        y1Max = int(h - patch_height)
        x1Max = int(w - patch_width)
        for y1 in range(0, y1Max):
            for x1 in range(0, x1Max):
                cropped = SegCrop(y1, x1, patch_height, patch_width, self.img, self.mask)
                self.crop_imgs.append(cropped)

        # print('Cropped ', len(self.crop_imgs), ' image crops.')
        return self.crop_imgs

    def crop_center(self, keep_height, keep_width):
        h = self.img.shape[0]
        w = self.img.shape[1]
        y1 = int((h - keep_height) / 2)
        x1 = int((w - keep_width) / 2)
        cropped = SegCrop(y1, x1, keep_height, keep_width, self.img, self.mask)
        self.crop_imgs.append(cropped)
        return self.crop_imgs

    def bin_image(self, pred_labels):
        pred_labels[pred_labels < 0.5] = 0
        pred_labels[pred_labels >= 0.5] = 1
        return pred_labels


class ImageSegDvnNetwork(object):

    def __init__(self, data_dir, img_size, dropout=0.75, learning_rate=0.01, inf_lr=50, binarize=False):

        self.sess = tf.InteractiveSession()
        self.sentinel = object()

        self.binarize = binarize
        self.input_size = img_size

        self.dropout = dropout
        self.learning_rate = learning_rate
        self.current_step = 0
        self.inf_lr = inf_lr

        self.mean = [0]
        self.std = [1]

        self.build_graph()

        # Initialize variables
        tf.global_variables_initializer().run()

        # Create a summary writer
        if data_dir:
            self.data_dir = data_dir
            self.writer = tf.summary.FileWriter('%s/log/train' % self.data_dir)
            self.val_writer = tf.summary.FileWriter('%s/log/val' % self.data_dir)
            self.saver = tf.train.Saver(tf.global_variables(),
                                        max_to_keep=50)
        else:
            self.data_dir = None
            self.writer = None
            self.val_writer = None
            self.saver = None

    def restore(self, path: object) -> object:
        """restore weights at `path`"""
        self.saver.restore(self.sess, path)

    def build_graph(self):
        """Build the model.

        It creates the network and all the operations needed for training and inference.
        Additionally it creates some ops for logging.

        Returns
        -------
        None
        """
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.images_pl = tf.placeholder(tf.float32, [None, self.input_size, self.input_size, 3])
        self.labels_pl = tf.placeholder(tf.float32, [None, self.input_size * self.input_size])
        self.keep_prob = tf.placeholder_with_default(self.dropout, shape=())
        self.gt_scores_pl = tf.placeholder(tf.float32, [None, 2])
        self.loss = 0

        self.raw_prediction = self.conv_net3(self.images_pl, self.labels_pl, self.keep_prob, None, True)
        self.predicted_values = tf.sigmoid(self.raw_prediction)

        self.gt_labels_pl = tf.placeholder(tf.float32, [None, self.input_size * self.input_size])
        self.eval_loss = tf.losses.cosine_distance(self.gt_labels_pl, self.labels_pl, axis=0)
        self.eval_gradient = tf.gradients(self.eval_loss, self.labels_pl)[0]

        ls = self.gt_scores_pl[:, 1]
        self.loss += tf.nn.sigmoid_cross_entropy_with_logits(logits=self.raw_prediction,
                                                             labels=self.gt_scores_pl[:, 1])

        # Gradient for generating adversarial examples
        self.adv_gradient = tf.gradients(self.loss, self.labels_pl)[0]

        # Gradient for inference (maximizing predicted value)
        self.gradient = tf.gradients(self.predicted_values, self.labels_pl)[0]

        self.sum_loss = tf.reduce_mean(self.loss)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        # optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self.sum_loss, global_step=self.global_step)

    def construct_conv2d(self, x, W, b, strides):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        tf.nn.relu(x)

    # Create the neural network
    def conv_net3(self, x_input, y_label, dropout_prob, reuse, is_training):
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):
            # TF Estimator input is a dict, in case of multiple inputs

            #nonlinearity = tf.nn.softplus
            nonlinearity = tf.nn.relu

            # 5x5 conv, 1 input, 32 outputs
            wc1_img = tf.Variable(tf.random_normal([5, 5, 3, 64], stddev=np.sqrt(2.0 / (5*5*3))))
            wc1_msk = tf.Variable(tf.random_normal([5, 5, 1, 64], stddev=np.sqrt(2.0 / (5*5*1))))

            # 5x5 conv, 32 inputs, 64 outputs
            wc2 = tf.Variable(tf.random_normal([5, 5, 64, 128], stddev=np.sqrt(2.0 / (5 * 5 * 64))))

            # 5x5 conv, 128 inputs, 128 outputs
            wc3 = tf.Variable(tf.random_normal([5, 5, 128, 128], stddev=np.sqrt(2.0 / (5*5*128))))

            # 3x3 conv, 32 inputs, 64 outputs
            wd1 = tf.Variable(tf.random_normal([6 * 6 * 128, 384], stddev=np.sqrt(2.0 / (6*6*128))))

            wd2 = tf.Variable(tf.random_normal([384, 192], stddev=np.sqrt(2.0 / 384)))

            # 1024 inputs, 10 outputs (class prediction)
            outw = tf.Variable(tf.random_normal([192, 1], stddev=np.sqrt(2.0 / 192)))

            bc1 = tf.Variable(tf.random_normal([64]))
            bc2 = tf.Variable(tf.random_normal([128]))
            bc3 = tf.Variable(tf.random_normal([128]))
            bd1 = tf.Variable(tf.random_normal([384]))
            bd2 = tf.Variable(tf.random_normal([192]))
            outb = tf.Variable(tf.random_normal([1]))

            y_square_lbs = tf.reshape(y_label, shape=[-1, self.input_size, self.input_size, 1])

            # Convolution Layer
            # conv1 = conv2d(x, weights['wc1'], biases['bc1'])
            xc1_1 = tf.nn.conv2d(x_input, wc1_img, strides=[1, 1, 1, 1], padding='SAME')
            xc1_2 = tf.nn.conv2d(y_square_lbs, wc1_msk, strides=[1, 1, 1, 1], padding='SAME')
            xc = tf.nn.bias_add(tf.add(xc1_1, xc1_2), bc1)  # tf.nn.bias_add(xc1_1 + xc1_2, bc1)
            self.conv1 = nonlinearity(xc)

            # Convolution Layer
            xc2 = tf.nn.conv2d(self.conv1, wc2, strides=[1, 2, 2, 1], padding='SAME')
            xc2 = tf.nn.bias_add(xc2, bc2)
            self.conv2 = nonlinearity(xc2)

            # Convolution Layer
            xc3 = tf.nn.conv2d(self.conv2, wc3, strides=[1, 2, 2, 1], padding='SAME')
            xc3 = tf.nn.bias_add(xc3, bc3)
            self.conv3 = nonlinearity(xc3)

            # Fully connected layer
            # Reshape conv2 output to fit fully connected layer input
            fc1 = tf.reshape(self.conv3, [-1, wd1.get_shape().as_list()[0]])
            fc1 = tf.add(tf.matmul(fc1, wd1), bd1)
            self.fc1 = nonlinearity(fc1)

            # Apply Dropout
            fc1 = tf.nn.dropout(fc1, dropout_prob)

            fc2 = tf.add(tf.matmul(fc1, wd2), bd2)
            self.fc2 = nonlinearity(fc2)

            # Output, class prediction
            out = tf.add(tf.matmul(fc2, outw), outb)
            self.out2 = tf.squeeze(out, axis=1)

        return self.out2

    def train(self, train_features, train_labels, epochs=1000, batch_size=20, train_ratio=0.9,
              test_feature=None, test_labels=None,
              trn_images=None, val_images=None, tst_images=None):

        """Train the value net.

        Parameters
        ----------
        train_features
            features used for training/validation
        train_labels
            the corresponding labels
        epochs : int
            the number of epochs to train for
        batch_size : int
            batch size to use
        train_ratio : float
            defines the ratio of the data used for training. The rest is used for validation

        Returns
        -------
        f1 score on validation at the end of training

        """

        print('Training parameters:\n')
        print('Mini-batch size =',batch_size)
        print('Learning rate =',self.learning_rate)
        print('Optimizer =','SGD')
        print('Example generation =','Inference + ?')

        print('\nInference parameters:')
        print('Inference step size = ', self.inf_lr)
        print('Num of Step = ', 30)

        train_features = np.array(train_features, np.float32)
        self.mean = np.mean(train_features, axis=0)
        self.std = np.std(train_features, axis=0) + 10 ** -6
        train_features -= self.mean
        train_features /= self.std

        # Split of some validation data
        if not hasattr(self, 'indices'):  # use existing splits if there are
            np.random.seed(0)
            self.indices = np.random.permutation(np.arange(len(train_features)))
        split_idx = int(len(train_features) * train_ratio)
        val_features = train_features[self.indices[split_idx:]]
        val_labels = train_labels[self.indices[split_idx:]]
        train_features = train_features[self.indices[:split_idx]]
        train_labels = train_labels[self.indices[:split_idx]]

        # Start training
        for epoch in range(0, epochs):
            start = time.time()
            print('----------------------------------------------------Starting epoch %d (it: %d)' % (epoch, self.current_step))
            sys.stdout.flush()

            # Randomize order
            order = np.random.permutation(np.arange(len(train_features)))
            train_features = train_features[order]
            train_labels = train_labels[order]

            # Start threads to fill sample queue
            queue = self._generator_queue(train_features, train_labels, batch_size)
            while True:
                data = queue.get(timeout=1000)
                if data is not self.sentinel:
                    # Do a training step to learn to corrently score the solution (predicted labels)
                    features, pred_labels, f1_scs = data
                    self.current_step, _ = self.sess.run(
                            [self.global_step, self.train_step],
                            feed_dict={self.images_pl: features,
                                       self.labels_pl: pred_labels,
                                       self.gt_scores_pl: f1_scs})
                else:
                    break

            print("-----------------------Epoch took %.2f seconds" % (time.time() - start))

            if epoch % 1 == 0 and self.saver:
                self.saver.save(self.sess, '%s/weights' % self.data_dir, global_step=self.current_step)

            self.predict_as_validation(train_features, train_labels)

            if tst_images is not None:
                if epoch % 1 == 0:
                    self.inferece_on_image_crops(tst_images)

            sys.stdout.flush()

        return None

    def inferece_on_image_crops(self, image_instances, dumpGIF=False):

        start_time = time.time()

        if image_instances is not None:

            test_mean_f = []

            for img_inst in image_instances:
                crops = img_inst.crop_imgs
                pred_full = []
                pred_denom = []

                #hist_lbls = []

                # crop
                all_crop_features, _ = img_inst.get_all_crop_imgs()
                prd_lbls0, hist0 = self.predict_with_history(all_crop_features)

                for crop_idx in range(0, len(crops)):
                    crop = crops[crop_idx]

                    pred_lbl = prd_lbls0[crop_idx]

                    crop.pred_labels = pred_lbl
                    fillsz_pred = crop.direct_fillback(crop.pred_labels)
                    fullsz_deno = crop.get_allone_patch()

                    # ImageSegDvnNetwork.dump_bmp(img_inst, pred_lbl=fillsz_pred, extra_name=str(crop_idx))

                    pred_full.append(fillsz_pred) # (bin_image(fillsz_pred))
                    pred_denom.append(fullsz_deno)

                sum_pred = np.sum(pred_full, axis=0)
                sum_deno = np.sum(np.array(pred_denom), axis=0)


                sum_deno[sum_deno == 0] = 1
                img_inst.pred_lbl = sum_pred / sum_deno

                full_pred = bin_image(img_inst.pred_lbl.flatten())
                full_mask = bin_image(img_inst.mask.flatten())

                ff1 = f1_score(full_mask, full_pred)
                precision, recall, f1_sc, true_sum = precision_recall_fscore_support(full_mask, full_pred)
                test_mean_f.append(ff1)

                # self.dump_gif(img_inst, hist_lbls, 24)
                ImageSegDvnNetwork.dump_bmp(img_inst)
                ImageSegDvnNetwork.dump_bmp(img_inst, pred_lbl=full_pred, extra_name='-bin')
                ImageSegDvnNetwork.dump_bmp(img_inst, pred_lbl=full_mask, extra_name='-msk')

            print("Test mean F1 = %.3f" % (np.mean(test_mean_f)))
            print("------------------Testing time %.2f seconds" % (time.time() - start_time))

    @staticmethod
    def dump_gif(img_instance, hist_lbls, img_sz):

        # print(hist_lbls[0].shape)

        gif_path = img_instance.data_dir + img_instance.fname + '.gif'

        gif_imgs = []
        for i in range(0, len(hist_lbls)):
            hl = hist_lbls[i][0] * 255
            #print(hl)
            lbl1d = np.array(hl, dtype=np.uint8)
            mask2d = lbl1d.reshape([img_sz, img_sz])
            gif_imgs.append(mask2d)

        #print('length=', len(gif_imgs))
        #print(hist_lbls.shape)

        imageio.mimsave(gif_path, gif_imgs, duration=0.1)

    @staticmethod
    def dump_bmp(img_instance, pred_lbl=None, extra_name=''):

        if pred_lbl is None:
            pred_lbl = img_instance.pred_lbl

        gif_path = img_instance.data_dir + img_instance.fname + '-pred' + extra_name + '.bmp'

        mask_cp = np.copy(pred_lbl)
        hl = mask_cp * 255
        lbl1d = np.array(hl, dtype=np.uint8)
        mask2d = lbl1d.reshape([img_instance.img.shape[0], img_instance.img.shape[1]])

        imageio.imsave(gif_path, mask2d)


    def predict_as_validation(self, features, gt_labels):
        """Run inference to obtain examples."""
        init_labels = self.get_initialization(features)
        pred_labels, history_preds = self.inference(features, init_labels, gt_labels=gt_labels,
                                                    pace_size=self.inf_lr, validation=True)

        # Score the predicted labels and return the labels and scores
        labels = np.zeros((gt_labels.shape[0], 2))
        labels[:, 1] = [self.gt_value(pred_labels[idx], gt_labels[idx], train=False) for idx in
                        np.arange(0, gt_labels.shape[0])]
        labels[:, 0] = 1 - labels[:, 1]

        if True:

            gt_values = np.zeros((gt_labels.shape[0], 2))
            gt_values[:, 1] = [self.gt_value(pred_labels[idx], gt_labels[idx], train=False) for idx in
                            np.arange(0, gt_labels.shape[0])]
            gt_values[:, 0] = 1 - gt_values[:, 1]
            print('sum_loss = ', self.sess.run(self.sum_loss,
                                               feed_dict={self.images_pl: features,
                                                          self.labels_pl: pred_labels,
                                                          self.gt_scores_pl: gt_values }))

            pred_labels[pred_labels < 0.5] = 0
            pred_labels[pred_labels >= 0.5] = 1

            test_mean_f = []
            for j in range(0, len(gt_labels)):
                ff1 = f1_score(gt_labels[j], pred_labels[j])
                test_mean_f.append(ff1)
            print("Train mean F1 = %.3f" % (np.mean(test_mean_f)))

    def generate_examples(self, features, gt_labels, train=False, val=False):
        """Run inference to obtain examples."""

        init_labels = self.get_initialization(features)
        is_advser = np.zeros(features.shape[0])

        # In training: Generate adversarial examples 50% of the time
        if train and np.random.rand() >= 0.5:
            # 50%: Start from GT; rest: start from zeros
            gt_indices = np.random.rand(gt_labels.shape[0]) > 0.5
            init_labels[gt_indices] = gt_labels[gt_indices]
            is_advser[gt_indices] = 1
            pred_labels, _ = self.inference(features, init_labels, gt_labels=gt_labels, num_iterations=1,
                                            pace_size=self.inf_lr, validation=val, adviseral=True)
            log = False

        # Otherwise: Run standard inference
        else:
            pred_labels, _ = self.inference(features, init_labels, gt_labels=gt_labels,num_iterations=30,
                                            pace_size=self.inf_lr, validation=val, adviseral=False)
            log = True

        # Score the predicted labels and return the labels and scores
        labels = np.zeros((gt_labels.shape[0], 2))
        labels[:, 1] = [self.gt_value(pred_labels[idx], gt_labels[idx], train=train) for idx in
                        np.arange(0, gt_labels.shape[0])]
        labels[:, 0] = 1 - labels[:, 1]

        return pred_labels, labels


    def gt_value(self, pred_lbs, gt_lbs, train=True):
        """Compute the ground truth value of some predicted labels."""

        pred_labels = pred_lbs.flatten() #np.reshape(pred_lbs,
        gt_labels = gt_lbs.flatten()

        if not train or self.binarize:
            pred_labels = np.array(pred_labels >= 0.5, np.float32)

        intersect = np.sum(np.min([pred_labels, gt_labels], axis=0))
        union = np.sum(np.max([pred_labels, gt_labels], axis=0))
        return 2 * intersect / float(intersect + max(10 ** -8, union))

    # def reduce_learning_rate(self, factor=0.1):
    #     """Reduce the current learing rate by multipling it with `factor`"""
    #     self.learning_rate *= factor
    #     self.train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(tf.reduce_mean(self.loss),
    #                                                                                      global_step=self.global_step)

    def _generator_queue(self, train_features, train_labels, batch_size, num_threads=5):
        que = queue.Queue(maxsize=20)

        # Build indices queue to ensure unique use of each batch
        indices_queue = queue.Queue()
        for idx in np.arange(0, len(train_features), batch_size):
            indices_queue.put(idx)

        def generate():
            try:
                while True:
                    # Get a batch
                    idx = indices_queue.get_nowait()
                    features = train_features[idx:min(len(train_features), idx + batch_size)]
                    gt_labels = train_labels[idx:min(len(train_labels), idx + batch_size)]

                    # Generate data (predcited labels and their true performance)
                    pred_labels, f1_scores = self.generate_examples(features, gt_labels, train=True, val=False)
                    que.put((features, pred_labels, f1_scores))
            except queue.Empty:
                que.put(self.sentinel)

        for _ in range(num_threads):
            thread = threading.Thread(target=generate)
            thread.start()

        return que

    def get_initialization(self, features):
        """Get the initial output hypothesis"""
        #if features.ndim == 1:
        #    features = features[None]

        init_lbs = np.zeros([features.shape[0], self.input_size * self.input_size])

        return init_lbs

    def predict(self, features, binarize=True):
        """
        Predict the labels for some features (single example).

        Parameters
        ----------
        features
        binarize : bool
            return the binarized results. If false: return scores

        Returns
        -------
        labels
            the predicted labels or scores
        """
        init_labels = self.get_initialization(features)
        features = np.array(features, np.float64)
        features -= self.mean[0]
        features /= self.std[0]

        labels, _ = self.inference(features,
                                   init_labels,
                                   pace_size=self.inf_lr,
                                   validation=True)#.flatten() #,).flatten()
                                   #num_iterations=30).flatten()

        # print(features.shape)
        # print(init_labels.shape)
        # print(labels.shape)

        if binarize:
            labels[labels < 0.5] = 0
            labels[labels >= 0.5] = 1
            return labels
        else:
            return labels


    def predict_with_history(self, features, binarize=True):

        init_labels = self.get_initialization(features)
        features = np.array(features, np.float64)
        features -= self.mean[0]
        features /= self.std[0]

        labels, hist = self.inference(features,
                                      init_labels,
                                      pace_size=self.inf_lr,
                                      num_iterations=30,
                                      validation=True)

        # if binarize:
        #     labels[labels < 0.5] = 0
        #     labels[labels >= 0.5] = 1

        return labels, hist

    def inference(self, features, initial_labels, gt_labels=None, pace_size=50, num_iterations=30,
                  validation=False, adviseral=False):

        history_outputs = []

        #pace_size = 50
        pred_labels = initial_labels

        # initial outputs
        history_outputs.append(np.copy(pred_labels))

        drop_prob = 0.75
        if validation:
            drop_prob = 1.0

        for idx in range(0, num_iterations):
            gradient = None
            # Compute the gradient
            if adviseral and not validation:
                gradient = self.sess.run(self.eval_gradient,
                                         feed_dict={self.labels_pl: pred_labels,
                                                    self.gt_labels_pl: gt_labels})

            else:
                gradient = self.sess.run(self.gradient,
                                         feed_dict={self.keep_prob: drop_prob,
                                                    self.images_pl: features,
                                                    self.labels_pl: pred_labels})



            # Update the labels to improve the predicted value
            #pred_labels += learning_rate * gradient
            pred_add = pace_size * gradient
            pred_labels += pred_add

            #print("step=",idx," grad_add=", pred_add)
            #print("step=",idx, " pred=", pred_labels)

            # Project back to the valid range
            pred_labels[np.isnan(pred_labels)] = 0
            pred_labels[pred_labels < 0] = 0
            pred_labels[pred_labels > 1] = 1

            # remember the history
            history_outputs.append(np.copy(pred_labels))

        return pred_labels, history_outputs

def crop_center(img, keep_height, keep_width):
    h = img.shape[0]
    w = img.shape[1]
    y1 = int((h - keep_height) / 2)
    y2 = int(y1 + keep_height)
    x1 = int((h - keep_width) / 2)
    x2 = int(x1 + keep_width)
    cropped = img[x1:x2,y1:y2]
    #print('crp=',cropped.shape)
    return cropped

def bin_image(pred_labels):
    pred_labels[pred_labels < 0.5] = 0
    pred_labels[pred_labels >= 0.5] = 1
    return pred_labels

def load_image_instances(picFolder, gtFolder, list_file, crop_count, crop_size, dumpFolder="horse_outpics/"):

    pics = []
    msks = []

    imgs = []

    patch_size = crop_size # default 24

    # create folder
    try:
        os.stat(dumpFolder)
    except:
        os.mkdir(dumpFolder)

    fileDir = os.path.dirname(os.path.realpath('__file__'))
    filename = os.path.join(fileDir, list_file)

    with open(filename, 'r') as openfileobject:
        for line1 in openfileobject:
            arr = line1.split()
            fname = 'horse-' + str(arr[1])
            imgpath = picFolder + 'horse-' + str(arr[1]) + '.bmp'
            muskpath = gtFolder + 'musk-' + str(arr[1]) + '.bmp'

            img = skimage.io.imread(imgpath, as_gray=False, conserve_memory=False)
            msk = skimage.io.imread(muskpath, as_gray=False, conserve_memory=False)

            # scale the value to range [0, 1]
            normalized_img = img / 255
            normalized_msk = bin_image(msk / 255)

            segImg = SegImage(dumpFolder, fname, patch_size, normalized_img, normalized_msk)
            if crop_count == 1:
                segImg.crop_center(patch_size, patch_size)
            else:
                # segImg.crop_image_full(patch_size, patch_size)
                segImg.crop_image_random(patch_size, patch_size, crop_count)

            imgCrops = segImg.crop_imgs
            for i in range(0, len(imgCrops)):
                crop = imgCrops[i]
                pics.append(crop.cropped_img)
                msks.append(crop.cropped_msk.flatten())

            imgs.append(segImg)

    print("Load image instances: " + str(len(pics)))

    parr = np.asarray(pics, dtype=np.float32)
    marr = np.asarray(msks, dtype=np.float32)
    print("Image shape: ",parr.shape)
    print("Label shape: ",marr.shape)
    return parr, marr, imgs


def evaluate_f1(predictor, features, labels):
    """Compute the F1 performance of a predictor on the given data."""
    mean_f = []
    for idx, (feature, lbl) in enumerate(zip(features, labels)):
        pred_lbl = predictor(np.asarray([feature], dtype=np.float32))[0]
        f1 = f1_score(lbl, pred_lbl)
        mean_f.append(f1)
        if idx % 20 == 0:
            print("%.3f (%d of %d)" % (np.mean(mean_f), idx, len(features)))
    print("%.3f" % (np.mean(mean_f)))

def run_horse():

    # Training Parameters

    picFolder = 'pics32/images/'
    gtFolder = 'pics32/musks/'

    trainfileName = 'pics32/train_list.txt'
    testfileName = 'pics32/test_list.txt'

    input_size = 24
    num_crops = 1 # the DVN paper use 36

    train_features, train_labels, trn_imgs = load_image_instances(picFolder, gtFolder, trainfileName, num_crops, input_size)
    test_features, test_labels, tst_imgs = load_image_instances(picFolder, gtFolder, testfileName, num_crops, input_size)

    weight_decay=0.01
    learning_rate=0.01
    inf_lr=50
    my_dropout = 0.75  # Dropout, probability to drop a unit
    num_epochs=300

    net = ImageSegDvnNetwork('./horse',
                             input_size,
                             dropout=my_dropout,
                             learning_rate=learning_rate,
                             inf_lr=inf_lr)

    net.train(train_features, train_labels, train_ratio=1.0, batch_size=20, epochs=num_epochs,
              test_feature=test_features, test_labels=test_labels,
              trn_images=trn_imgs, tst_images=tst_imgs)
    # net.reduce_learning_rate()
    #net.train(train_features, train_labels, train_ratio=0.95, epochs=int(num_epochs*0.5))
    #net.reduce_learning_rate()
    #net.train(train_features, train_labels, train_ratio=0.95, epochs=int(num_epochs*0.1))# / 2.0))

    # Train more, with the full training set
    #net.train(train_features, train_labels, train_ratio=1, epochs=int(num_epochs * .1))

    print('\n\nFnial test accuracy:\n')
    net.inferece_on_image_crops(tst_imgs)


if __name__=='__main__':
    run_horse()
