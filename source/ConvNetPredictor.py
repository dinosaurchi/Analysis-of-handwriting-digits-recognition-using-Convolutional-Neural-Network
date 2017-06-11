import numpy as np
import tensorflow as tf
import os
from datetime import datetime


class ConvNetPredictor:
    # All training data has to have the size of 28x28x1 and the total classes of input data has to be 10

    def __init__(self):
        self.highest_prob = 0.9
        self.img_size = 28
        self.img_size_flat = self.img_size * self.img_size
        self.img_shape = (self.img_size, self.img_size)
        self.total_classes = 10

        self.session = None
        # Store layers weight & bias
        self.weights = {
            # 5x5 conv, 1 input, 32 outputs
            'w_conv_1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
            # 5x5 conv, 32 inputs, 64 outputs
            'w_conv_2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
            # 5x5 conv, 64 inputs, 128 outputs
            'w_conv_3': tf.Variable(tf.random_normal([2, 2, 64, 128])),
            # fully connected, 7*7*64 inputs, 1024 outputs
            'w_full': tf.Variable(tf.random_normal([7 * 7 * 128, 1024])),
            # 1024 inputs, 10 outputs (class prediction)
            'w_out': tf.Variable(tf.random_normal([1024, self.total_classes]))
        }

        self.biases = {
            'b_conv_1': tf.Variable(tf.random_normal([32])),
            'b_conv_2': tf.Variable(tf.random_normal([64])),
            'b_conv_3': tf.Variable(tf.random_normal([128])),
            'b_full': tf.Variable(tf.random_normal([1024])),
            'b_out': tf.Variable(tf.random_normal([self.total_classes]))
        }

        self.x = tf.placeholder(tf.float32, [None, self.img_size_flat])
        self.y_true = tf.placeholder(tf.float32, [None, self.total_classes])
        self.y_true_label = tf.placeholder(tf.int64, [None])

        # keep_prob is the probability of keeping a neuron not being ignored in an iteration
        self.keep_prob = tf.placeholder(tf.float32)

        # Forward propagation for the input image 'x'
        self.predicted_y = self.forward(self.x, self.keep_prob)

        # convert the one-hot encoding into the related label
        self.predicted_y_label = tf.argmax(self.predicted_y, dimension=1)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.predicted_y, labels=self.y_true)

        # Average Cross-entropy
        self.cost = tf.reduce_mean(self.cross_entropy)

        self.correct_prediction = tf.equal(self.predicted_y_label, self.y_true_label)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def save_model(self, acc, batch_size, i):
        saver = tf.train.Saver()
        dir_path = './model_cnn-i' + str(i) + '-b' + str(batch_size) + '-a' + str(acc * 100)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_path = saver.save(self.session, dir_path + "/model.ckpt")
        print '[' + str(datetime.now()) + '] ' + "Model saved in file: %s" % save_path

    def restore_model(self, model_dir_path):
        saver = tf.train.Saver()
        self.session = tf.Session()
        saver.restore(self.session, model_dir_path + '/model.ckpt')

        return self.session

    @staticmethod
    def pooling(x, kernel_size, strds):
        # It will look for the maximum value within a (kernel_size x kernel_size) window
        # For instance:
        #           x = [
        #                   [1., 2.,  3.,  4.],
        #                   [5., 6.,  7.,  8.],
        #                   [9., 10., 11., 12.]
        #               ]
        # => max_pool (VALID padding) with kernel size = 2 and stride_x = stride_y = 1 (step of sliding window)
        #  ....returns the values:
        #               [
        #                   [  6.,  7.,   8.],
        #                   [  10., 11.,  12.]
        #               ]
        # => max_pool (VALID padding) with kernel size = 2 and stride_x = stride_y = 2 (step of sliding window)
        #  ....returns the values:
        #               [
        #                   [  6.,        8.],
        #               ]
        # It means valid padding algorithm only consider the complete kernel, not half of kernel or something else
        # ... (which is happened if the size of x is not divisible by the kernel size)
        #
        # In order to make the incomplete kernel being involved, we use SAME padding
        #
        # => max_pool (SAME padding) with kernel size = 2 and stride_x = stride_y = 1 (step of sliding window)
        #  ....returns the values:
        #               [
        #                   [  6.,  7.,   8.   8.],
        #                   [  10., 11.,  12., 12.],
        #                   [  10., 11.,  12.  12.]
        #               ]
        # => max_pool (SAME padding) with kernel size = 2 and stride_x = stride_y = 2 (step of sliding window)
        #  ....returns the values:
        #               [
        #                   [  6.,        8.],
        #                   [  10.        12.]
        #               ]

        return tf.nn.avg_pool(x,
                              ksize=[1, kernel_size, kernel_size, 1],
                              strides=[1, strds, strds, 1],
                              padding='SAME')

    @staticmethod
    def convolve_2d(x, W, b, strds):
        x = tf.nn.conv2d(x, W, strides=[1, strds, strds, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)

        # It is equivalent to compute 'max(a, 0) for a in x'
        # relu == rectified linear unit
        return tf.nn.relu(x)

    def forward(self, x, dropout):
        # -1 stands for automatically implication of size
        # We will have a tensor of size mx28x28x1 where m is total number of training
        # ... examples
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        kernel_size = 2

        # Convolution Layer 1
        convolutional_layer_1 = self.convolve_2d(x, self.weights['w_conv_1'], self.biases['b_conv_1'], 1)
        convolutional_layer_1 = self.pooling(convolutional_layer_1, kernel_size, 2)

        # Convolution Layer 2
        convolutional_layer_2 = self.convolve_2d(convolutional_layer_1, self.weights['w_conv_2'],
                                                 self.biases['b_conv_2'], 1)
        convolutional_layer_2 = self.pooling(convolutional_layer_2, kernel_size, 1)

        # Convolution Layer 3
        convolutional_layer_3 = self.convolve_2d(convolutional_layer_2, self.weights['w_conv_3'],
                                                 self.biases['b_conv_3'], 1)
        convolutional_layer_3 = self.pooling(convolutional_layer_3, kernel_size, 2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fully_connected_layer = tf.reshape(convolutional_layer_3, [-1, self.weights['w_full'].get_shape().as_list()[0]])
        fully_connected_layer = tf.add(tf.matmul(fully_connected_layer, self.weights['w_full']), self.biases['b_full'])

        fully_connected_layer = tf.nn.relu(fully_connected_layer)
        # Apply Dropout
        fully_connected_layer = tf.nn.dropout(fully_connected_layer, dropout)

        # Output, class prediction
        output_layer = tf.add(tf.matmul(fully_connected_layer, self.weights['w_out']), self.biases['b_out'])

        return output_layer

    def train(self,
              train_data,
              test_data,
              learning_rate=0.001,
              iterations=500000,
              keeping_probability=0.75,
              minimum_savable_probability=0.9):

        if train_data is not None and test_data is not None:
            print "Start training"

            self.highest_prob = minimum_savable_probability

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
            # We can try with different batch sizes
            for batch_size in [128]:
                self.session = tf.Session()
                self.session.run(tf.global_variables_initializer())
                i = 1

                # Train until the max iterations
                while i < iterations:
                    x_batch, y_batch = train_data.next_batch(batch_size)
                    # Run optimization (back-propagation because we use CNN) with keeping probability is 0.75
                    self.session.run(optimizer,
                                     feed_dict={
                                         self.x: x_batch,
                                         self.y_true: y_batch,
                                         self.keep_prob: keeping_probability})

                    if i % 10 == 0:
                        print '[' + str(datetime.now()) + '] ' \
                              + "Progress : " + str(i + 1) \
                              + " / " + str(iterations) \
                              + ' (batch size = ' + str(batch_size) + ')'

                    if i % 50 == 0:
                        # convert the one-hot encoding into the related label
                        batch_y_label = np.array([label.argmax() for label in y_batch])

                        # When do testing, we assign dropout probability to 0, means the keeping
                        # ... probability is 1
                        loss, acc = self.session.run([self.cost, self.accuracy],
                                                     feed_dict={
                                                         self.x: x_batch,
                                                         self.y_true: y_batch,
                                                         self.y_true_label: batch_y_label,
                                                         self.keep_prob: 1.})
                        print '[' + str(datetime.now()) + '] ' + "Current minibatch Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{0:.1%}".format(acc)

                    if i % 1000 == 0:
                        # When do testing, we assign dropout probability to 0, means the keeping
                        # ... probability is 1
                        real_labels = np.array([label.argmax() for label in test_data.labels])
                        loss, acc = self.session.run([self.cost, self.accuracy],
                                                     feed_dict={
                                                         self.x: test_data.images[0:10000],
                                                         self.y_true: test_data.labels[0:10000],
                                                         self.y_true_label: real_labels,
                                                         self.keep_prob: 1.})
                        print '[' + str(datetime.now()) + '] ' + "Current testing data Loss= " + \
                              "{:.6f}".format(loss) + ", Training Accuracy= " + \
                              "{0:.1%}".format(acc)

                        if acc > highest_prob:
                            print '[' + str(datetime.now()) + '] ' + 'New highest probability : ' + str(acc * 100) + '%'
                            highest_prob = acc
                            self.save_model(acc, batch_size, i)

                        print 'Current highest : ' + str(highest_prob * 100) + '%'

                    i += 1

                print '-----------------------------------------------------------'
                # When do testing, we assign dropout probability to 0, means the keeping
                # ... probability is 1
                real_labels = np.array([label.argmax() for label in test_data.labels])
                loss, acc = self.session.run([self.cost, self.accuracy],
                                             feed_dict={
                                                 self.x: test_data.images,
                                                 self.y_true: test_data.labels,
                                                 self.y_true_label: real_labels,
                                                 self.keep_prob: 1.})
                print '[' + str(datetime.now()) + '] ' + "Test-set minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{0:.1%}".format(acc)

                if acc > highest_prob:
                    highest_prob = acc
                    print '[' + str(datetime.now()) + '] ' + 'New highest probability : ' + str(acc * 100) + '%'
                    self.save_model(acc, batch_size, i)

    def predict(self, x):
        return self.session.run([self.predicted_y_label],
                                feed_dict={
                                    self.x: x,
                                    self.keep_prob: 1.
                                })

    def run_test(self, test_data):
        classes = np.array([label.argmax() for label in test_data.labels])
        loss, acc = self.session.run([self.cost, self.accuracy],
                                     feed_dict={
                                         self.x: test_data.images,
                                         self.y_true: test_data.labels,
                                         self.y_true_label: classes,
                                         self.keep_prob: 1.})
        return loss, acc
