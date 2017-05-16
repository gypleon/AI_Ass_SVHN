from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

from six.moves import xrange
import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


# configuration
BATCH_SIZE = 50
NUM_SUBPLOT_COLS = 10
DATASET_PATH = "./data/train_32x32.mat"

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

class DataLoader:
  def __init__(self, data_path, batch_size=50):
    data = sio.loadmat(data_path)
    self.batch_size = batch_size
    self.images = data['X']
    self.labels = data['y']
    self.images = np.transpose(self.images, (3, 0, 1, 2))
    
    # input queue
    examples = zip(self.labels, self.images)
    self.example_queue = tf.train.input_producer(examples)
    '''
    num_samples = self.labels.shape[0]
    num_batches = num_samples // batch_size
    # last_batch_size = num_samples - num_batches * batch_size
    split_conf = [batch_size*(i+1) for i in range(num_batches)]
    split_conf.append(num_samples)

    self.image_batches = np.split(np.transpose(self.images, (3, 0, 1, 2)), split_conf, axis=0)
    self.label_batches = np.split(self.labels, split_conf, axis=0)
    '''

  def preprocess(self, image):
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)
    float_image = tf.image.per_image_standardization(distorted_image)
    return float_image

  def load_batch(self):
    example = self.example_queue.dequeue()
    image = tf.stack(example[1])
    label = tf.stack(example[0])
    image = self.preprocess(image)
    image.set_shape([32, 32, 3])
    label.seg_shape([1])
    image_batch, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=self.batch_size,
      num_threads=4,
      capacity=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + 3 * self.batch_size,
      min_after_dequeue=NUM_EXAMPLES_PER_EPOCH_FOR_EVAL) 
    tf.summary.image('images', images)
    return image_batch, tf.reshape(label_batch, [self.batch_size])


if __name__ == "__main__":

  dataloader = DataLoader(DATASET_PATH, BATCH_SIZE)

  batch_count = 0
  fig = plt.figure()
  num_plot_cols = NUM_SUBPLOT_COLS
  num_plot_rows = int(math.ceil(BATCH_SIZE/num_plot_cols))
  labels = []
  for image_batch, label_batch in dataloader.iter():
    if batch_count > 0:
      break
    for batch_i in range(BATCH_SIZE):
      sub_plot = fig.add_subplot(num_plot_rows, num_plot_cols, batch_i+1)
      plt.imshow(image_batch[batch_i])
    batch_count += 1
    labels.append([label[0] for label in label_batch])

  print(labels)
  plt.show()
