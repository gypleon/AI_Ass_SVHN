from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import math

import tensorflow as tf
import numpy as np

from dataloader import DataLoader
import model

FLAGS = tf.app.flags.FLAGS

# configuration
tf.app.flags.DEFINE_string ("test_set_path", "../data/test_images.mat", "path of the test set")
tf.app.flags.DEFINE_string ("log_dir", "./trained_model", "path of checkpoints/logs")
tf.app.flags.DEFINE_integer('num_test_examples', 1000, "")


# test model
def main(_):
  if not tf.gfile.Exists(FLAGS.log_dir):
    raise Exception("trained model missing.")

  with tf.Graph().as_default():
    test_loader = DataLoader(FLAGS.test_set_path, num_valid_samples=FLAGS.num_test_examples, is_test=True)
    images = test_loader.load_batch()

    with tf.variable_scope("svhn"):
      logits = model.inference(images)
      prediction = tf.nn.top_k(logits)
    
    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as session:
      test_loader.load(session)
      ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        session.run(tf.global_variables_initializer())
        print(ckpt.model_checkpoint_path, "loaded")
      else:
        raise Exception("did not find checkpoint on", FLAGS.log_dir)
      labels = session.run([prediction])
      print(labels)
      labels[labels==0] = 10
      print(labels)
      test_loader.close(session)

if __name__ == "__main__":
  tf.app.run()
