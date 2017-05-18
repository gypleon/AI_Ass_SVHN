from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import threading

from six.moves import xrange
import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf


# configuration
BATCH_SIZE = 128
NUM_SUBPLOT_COLS = 10
DATASET_PATH = "../data/train_32x32.mat"
VALID_DATASET_PATH = "./data/test_32x32.mat"
GEN_TEST_PATH = "../data/test_images.mat"
CROP_RATE = 0.75
CROP_H = int(32 * CROP_RATE)
CROP_W = int(32 * CROP_RATE)

NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 73257
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100

class DataLoader:
  def __init__(self, data_path, batch_size=50, num_valid_samples=None):
    print("loading raw file:", data_path)
    data = sio.loadmat(data_path)
    self.images = data['X']
    self.labels = data['y']
    self.images = np.transpose(self.images, (3, 0, 1, 2))
    self.labels[self.labels==10] = 0
    self.num_valid_samples = num_valid_samples
    self.batch_size = num_valid_samples or batch_size
    if self.num_valid_samples != None:
      self.random_valid_set()
    # create queue
    print("filling input queue")
    self.queue_image = tf.placeholder(tf.int32, shape=[self.batch_size, 32, 32, 3], name="input_images")
    self.queue_label = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name="input_labels")
    if self.num_valid_samples == None:
      capacity=int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4) + 3 * self.batch_size
    else:
      capacity=int(self.num_valid_samples * 0.4) + 3 * self.batch_size
    self.example_queue = tf.FIFOQueue(
      capacity=capacity,
      dtypes=[tf.int32, tf.int32],
      shapes=[[32, 32, 3], [1]])
    self.enqueue = self.example_queue.enqueue_many([self.queue_image, self.queue_label])
    self.enqueue_thread = None
    self.coord = tf.train.Coordinator()
    self.threads = None

  def random_valid_set(self):
    num_valid_samples = self.num_valid_samples
    dataset_size = self.images.shape[0]
    start = np.random.randint(0, dataset_size - num_valid_samples)
    self.images = self.images[start:start+num_valid_samples]
    self.labels = self.labels[start:start+num_valid_samples]
    print("random validation set [%d:%d]" % (start, start+num_valid_samples))
    
  def data_stream(self, session):
    try:
      start = 0
      dataset_size = len(self.labels)
      while not self.coord.should_stop():
        end = start + self.batch_size
        # print("loading [%d:%d] into input queue..." % (start, end))
        if end <= dataset_size:
          image_batch = self.images[start:end]
          label_batch = self.labels[start:end]
          start = end
        else:
          remaining = end - dataset_size
          image_batch = np.concatenate((self.images[start:dataset_size], self.images[0:remaining]))
          label_batch = np.concatenate((self.labels[start:dataset_size], self.labels[0:remaining]))
          start = remaining
        session.run(
          self.enqueue, 
          feed_dict={
            self.queue_image : image_batch,
            self.queue_label : label_batch})
      print("data stream closed.")
    except:
      print("data stream closed.")
      sys.exit()

  def preprocess(self):
    image, label = self.example_queue.dequeue()
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.int32)
    if self.num_valid_samples == None:
      image = tf.random_crop(image, [CROP_H, CROP_W, 3])
      image = tf.image.random_flip_left_right(image)
      image = tf.image.random_brightness(image, max_delta=63)
      image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    else:
      image = tf.image.resize_image_with_crop_or_pad(image, CROP_H, CROP_W)
    float_image = tf.image.per_image_standardization(image)
    return float_image, label

  def load_batch(self):
    image, label= self.preprocess()
    image.set_shape([int(32 * CROP_RATE), int(32 * CROP_RATE), 3])
    label.set_shape([1])
    if self.num_valid_samples == None:
      image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=self.batch_size,
        num_threads=4,
        capacity=int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4) + 3 * self.batch_size,
        min_after_dequeue=int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * 0.4))
    else:
      image_batch, label_batch = tf.train.batch(
        [image, label],
        batch_size=self.batch_size,
        num_threads=4,
        capacity=int(self.num_valid_samples * 0.4) + 3 * self.batch_size)
    # tf.summary.image('images', image_batch)
    print("loading batch of samples:", self.batch_size) 
    return image_batch, tf.reshape(label_batch, [self.batch_size])

  def load(self, session):
    self.enqueue_thread = threading.Thread(target=self.data_stream, args=[session])
    self.enqueue_thread.isDaemon()
    self.enqueue_thread.start()
    self.threads = tf.train.start_queue_runners(coord=self.coord, sess=session)
    
  def close(self, session):
    session.run(self.example_queue.close(cancel_pending_enqueues=True))
    self.coord.request_stop()
    self.coord.join(self.threads)
    print("dataloader closed successfully.")
