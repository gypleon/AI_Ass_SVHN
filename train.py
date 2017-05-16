import tensorflow as tf
import numpy as np
import time
import math

from dataloader import DataLoader
from model import SVHN

# configuration
BATCH_SIZE = 50
TRAIN_SET_PATH = "./data/train_32x32.mat"
MAX_EPOCHS = 100
LEARNING_RATE = 1.0


# train model
def main(_):
  seed = int(time.time())

  train_set_loader = DataLoader(TRAIN_SET_PATH, BATCH_SIZE)
  valid_set_loader = DataLoader(VALID_SET_PATH)
  
  with tf.Graph().as_default(), tf.Session() as session:
    images = tf.placeholder(tf.int64, shape=[batch_size, 32, 32], name="inputs")
    labels = tf.placeholder(tf.int64, shape=[batch_size], name="labels")

    tf.set_random_seed(seed)
    initializer = tf.contrib.layers.xavier_initializer_conv2d(seed=seed)
    global_step = tf.contrib.framework.get_or_create_global_step()

    with tf.variable_scope("SVHN", initializer=initializer):
      logits = SVHN.inference(images)
      loss = SVHN.loss(logits, labels)
      op = SVHN.optimize(loss, global_step)

    saver = tf.train.Saver()

    with tf.variable_scope("SVHN", reuse=True):
      logits = SVHN.inference(images)
      loss = SVHN.loss(logits, labels)
    
    tf.global_vairables_initializer().run()
    
    for epoch in range(MAX_EPOCHS):
      for image_batch, label_batch in train_set_loader.iter():
        session.run([], {})
        

if __name__ == "__main__":
  tf.app.run()
