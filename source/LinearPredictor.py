from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
import os
from datetime import datetime


class LinearPredictor:
    # All training data has to have the size of 28x28x1 and the total classes of input data has to be 10

    def __init__(self):
        self.highest_prob = 0.9
        self.img_size = 28
        self.img_size_flat = self.img_size * self.img_size
        self.img_shape = (self.img_size, self.img_size)
        self.total_classes = 10

        self.session = None
        # Store layers weight & bias

        self.weights = tf.Variable(tf.zeros([self.img_size_flat, self.total_classes]))
        self.biases = tf.Variable(tf.zeros([self.total_classes]))

        self.x = tf.placeholder(tf.float32, [None, self.img_size_flat])
        self.y_true = tf.placeholder(tf.float32, [None, self.total_classes])
        self.y_true_label = tf.placeholder(tf.int64, [None])

        self.logits = tf.matmul(self.x, self.weights) + self.biases
        # Convert to probability result
        self.predicted_y = tf.nn.softmax(self.logits)

        # convert the one-hot encoding into the related label
        self.predicted_y_label = tf.argmax(self.predicted_y, dimension=1)

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_true)

        # Average Cross-entropy
        self.cost = tf.reduce_mean(self.cross_entropy)

        self.correct_prediction = tf.equal(self.predicted_y_label, self.y_true_label)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def save_model(self, acc, batch_size, i):
        saver = tf.train.Saver()
        dir_path = './model_linear-i' + str(i) + '-b' + str(batch_size) + '-a' + str(acc * 100)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        save_path = saver.save(self.session, dir_path + "/model.ckpt")
        print '[' + str(datetime.now()) + '] ' + "Model saved in file: %s" % save_path

    def restore_model(self, model_dir_path):
        saver = tf.train.Saver()
        self.session = tf.Session()
        saver.restore(self.session, model_dir_path + '/model.ckpt')

        return self.session

    def train(self,
              train_data,
              test_data,
              learning_rate=0.001,
              iterations=500000,
              minimum_savable_probability=0.9):

        if train_data is not None and test_data is not None:
            print "Start training"

            self.highest_prob = minimum_savable_probability

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
            # We can try with different batch sizes
            for batch_size in [100, 500, 1000, 1200, 1350, 1572, 1864, 2121, 3344, 4677, 5120]:
                self.session = tf.Session()
                self.session.run(tf.global_variables_initializer())

                for i in range(iterations):
                    # If the next batch index is out of the total number training examples,
                    # ... it will go back to the position of the first example
                    x_batch, y_true_batch = train_data.next_batch(batch_size)

                    feed_dict_train = {self.x: x_batch,
                                       self.y_true: y_true_batch}
                    self.session.run(optimizer, feed_dict=feed_dict_train)

                    if i % 10 == 0:
                        print '[' + str(datetime.now()) + '] ' + "Progress : " + str(i + 1) + " / " + str(iterations) \
                              + ' (batch size = ' + str(batch_size) + ')'

                    if i % 200 == 0:
                        real_labels = np.array([label.argmax() for label in test_data.labels])
                        feed_dict_test = {self.x: test_data.images,
                                          self.y_true: test_data.labels,
                                          self.y_true_label: real_labels}
                        acc = self.session.run(self.accuracy, feed_dict=feed_dict_test)
                        print '[' + str(datetime.now()) + '] ' + "Accuracy on test-set: {0:.1%}".format(acc)

                        if acc > self.highest_prob:
                            print '[' + str(datetime.now()) + '] ' + 'New highest probabilty : ' + str(acc * 100) + '%'
                            self.highest_prob = acc
                            self.save_model(acc, batch_size, i)

                print '----------------------------------------'
                real_labels = np.array([label.argmax() for label in test_data.labels])
                feed_dict_test = {self.x: test_data.images,
                                  self.y_true: test_data.labels,
                                  self.y_true_label: real_labels}
                acc = self.session.run(self.accuracy, feed_dict=feed_dict_test)
                print '[' + str(datetime.now()) + '] ' + "Accuracy on test-set: {0:.1%}".format(acc)

                if acc > self.highest_prob:
                    print '[' + str(datetime.now()) + '] ' + 'New highest probabilty : ' + str(acc * 100) + '%'
                    self.save_model(acc, batch_size, i)

    def predict(self, x):
        return self.session.run([self.predicted_y_label],
                                feed_dict={
                                    self.x: x,
                                })

    def run_test(self, test_data):
        classes = np.array([label.argmax() for label in test_data.labels])
        loss, acc = self.session.run([self.cost, self.accuracy],
                                     feed_dict={
                                         self.x: test_data.images,
                                         self.y_true: test_data.labels,
                                         self.y_true_label: classes})
        return loss, acc
