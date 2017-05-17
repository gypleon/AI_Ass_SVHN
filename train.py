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
tf.app.flags.DEFINE_integer("batch_size", 100, "batch size")
tf.app.flags.DEFINE_integer("log_frequency", 10, "log frequency")
tf.app.flags.DEFINE_string ("train_set_path", "./data/train_32x32.mat", "path of the train set")
tf.app.flags.DEFINE_string ("log_dir", "/tmp/svhn/logs", "path of checkpoints/logs")
tf.app.flags.DEFINE_integer("max_epochs", 100000, "max number of epochs")
tf.app.flags.DEFINE_float  ("learning_rate", 1.0, "initial learning rate")
tf.app.flags.DEFINE_boolean('log_device_placement', False, "Whether to log device placement.")


# train model
def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)

  seed = int(time.time())

  with tf.Graph().as_default():
    dataloader = DataLoader(FLAGS.train_set_path, FLAGS.batch_size)
    images, labels = dataloader.load_batch()

    tf.set_random_seed(seed)
    initializer = tf.contrib.layers.xavier_initializer_conv2d(seed=seed)
    global_step = tf.contrib.framework.get_or_create_global_step()

    with tf.variable_scope("svhn", initializer=initializer):
      logits = model.inference(images)
      # logits = tf.Print(logits, [logits], message="TEST logits:", summarize=FLAGS.batch_size)
      # labels = tf.Print(labels, [labels], message="TEST labels:", summarize=FLAGS.batch_size)
      loss = model.loss(logits, labels)
      # loss = tf.Print(loss, [loss], message="TEST loss:", summarize=FLAGS.batch_size)
      op = model.optimize(loss, global_step)

    scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer())

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""
      def after_create_session(self, session, coord):
        dataloader.load(session)
        print("TEST after_create_session")

      def begin(self):
        # print("TEST begin")
        self._step = -1
        self._start_time = time.time()

      def end(self, session):
        print("TEST end")
        dataloader.close(session)

      def before_run(self, run_context):
        # print("TEST before_run")
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        # print("TEST after_run")
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print(format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        scaffold=scaffold,
        checkpoint_dir=FLAGS.log_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_epochs),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(op)

    '''
    saver = tf.train.Saver()

    with tf.variable_scope("SVHN", reuse=True):
      logits = SVHN.inference(images)
      loss = SVHN.loss(logits, labels)
    
    tf.global_vairables_initializer().run()
    
    for epoch in range(FLAGS.max_epochs):
      for image_batch, label_batch in dataloader.iter():
        session.run([], {})
    '''
        

if __name__ == "__main__":
  tf.app.run()
