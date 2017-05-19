from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import math

import tensorflow as tf
import numpy as np

import dataloader
from dataloader import DataLoader
import model

FLAGS = tf.app.flags.FLAGS

# configuration
tf.app.flags.DEFINE_string ("test_set_path", dataloader.GEN_TEST_PATH, "path of the test set")
tf.app.flags.DEFINE_string ("log_dir", "./trained_model", "path of checkpoints/logs")
tf.app.flags.DEFINE_string ("output_path", dataloader.GEN_LABELS_PATH, "path of checkpoints/logs")
tf.app.flags.DEFINE_integer('num_test_examples', 1000, "")


# test model
def main(_):
  if not tf.gfile.Exists(FLAGS.log_dir):
    raise Exception("trained model missing.")
  if not tf.gfile.Exists(FLAGS.test_set_path):
    raise Exception("test set missing.")

  with tf.Graph().as_default():
    test_loader = DataLoader(FLAGS.test_set_path, num_valid_samples=FLAGS.num_test_examples, is_test=True)
    test_images = test_loader.load_batch()

    with tf.variable_scope("svhn"):
      logits = model.inference(test_images)
      predictions = model.prediction(logits)
    
    variable_averages = tf.train.ExponentialMovingAverage(model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    options = tf.RunOptions(timeout_in_ms=10000)
    with tf.Session() as session:
      session.run(tf.global_variables_initializer())
      test_loader.load(session)
      ckpt = tf.train.get_checkpoint_state(FLAGS.log_dir)
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(session, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path, "loaded")
      else:
        raise Exception("did not find checkpoint on", FLAGS.log_dir)
      preds = session.run(predictions, options=options)
      with open(FLAGS.output_path, "w") as f:
        for pred in preds:
          if 0 == pred[0][0]:
            f.write("%d\n" % 10)
          else:
            f.write("%d\n" % pred[0][0])
      test_loader.close(session)

if __name__ == "__main__":
  tf.app.run()
