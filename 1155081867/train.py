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
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.app.flags.DEFINE_integer("valid_batch_size", 1000, "validation batch size")
tf.app.flags.DEFINE_integer("log_frequency", 10, "log frequency")
tf.app.flags.DEFINE_integer("eval_frequency", 200, "evaluation frequency")
tf.app.flags.DEFINE_string ("train_set_path", "../data/train.mat", "path of the train set")
tf.app.flags.DEFINE_string ("valid_set_path", "../data/test_32x32.mat", "path of the test set")
tf.app.flags.DEFINE_string ("log_dir", "./trained_model", "path of checkpoints/logs")
tf.app.flags.DEFINE_integer("max_steps", 100000, "max number of steps (batchs)")
tf.app.flags.DEFINE_float  ("learning_rate", 0.1, "initial learning rate")
tf.app.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")
tf.app.flags.DEFINE_integer('num_valid_examples', 1000, "")


# train model
def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  seed = int(time.time())

  with tf.Graph().as_default():
    train_loader = DataLoader(FLAGS.train_set_path, FLAGS.batch_size)
    valid_loader = DataLoader(FLAGS.valid_set_path, num_valid_samples=FLAGS.num_valid_examples)
    train_images, train_labels = train_loader.load_batch()
    valid_images, valid_labels = valid_loader.load_batch()

    tf.set_random_seed(seed)
    global_step = tf.contrib.framework.get_or_create_global_step()

    with tf.variable_scope("svhn"):
      logits = model.inference(train_images)
      loss = model.loss(logits, train_labels)
      train_op = model.optimize(loss, global_step, FLAGS.learning_rate, FLAGS.batch_size)

    with tf.variable_scope("svhn", reuse=True):
      logits = model.inference(valid_images)
      top_k_op = tf.nn.in_top_k(logits, valid_labels, 1)
    
    scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())
    saver = tf.train.Saver(tf.trainable_variables())

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""
      def after_create_session(self, session, coord):
        train_loader.load(session)
        valid_loader.load(session)

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def end(self, session):
        train_loader.close(session)
        valid_loader.close(session)

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          if self._step % FLAGS.eval_frequency == 0:
            precision = model.evaluate(run_context.session, top_k_op, FLAGS.num_valid_examples)
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch), precision = %.2f%%')
            print(format_str % (datetime.now(), self._step, loss_value,
                                 examples_per_sec, sec_per_batch, precision))
          else:
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), self._step, loss_value,
                                 examples_per_sec, sec_per_batch))


    with tf.train.MonitoredTrainingSession(
        scaffold=scaffold,
        checkpoint_dir=FLAGS.log_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook(),
               tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.log_dir, saver=saver, save_steps=100)],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)

if __name__ == "__main__":
  tf.app.run()
