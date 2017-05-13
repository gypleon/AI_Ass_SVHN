import tensorflow as tf
import numpy as np
import time
import math

from dataloader import DataLoader

# configuration
BATCH_SIZE = 50
TRAIN_SET_PATH = "./data/train_32x32.mat"
MAX_EPOCHS = 100
LEARNING_RATE = 0.01

# define model
def conv2d(input_data, output_dim, width, height, scope):
  with tf.variable_scope(scope):
    weights = tf.get_variable()
    bias = tf.get_variable()
  return tf.nn.conv2d(input_data, weights, strides=[], padding="VALID") + bias

def fc(input_data, output_dim, scope):
  shape = intput_data.get_shape().as_list()
  with tf.variable_scope(scope):
    weights = tf.get_variable()
    bias = tf.get_variable()
  return tf.matmul() + bias

def loss(input_data, scope):
  with tf.variable_scope(scope):
    
  return 

def update(loss, ):
  with tf.variable_scope(scope):
    learning_rate = tf.get_variable(learning_rate, trainable=False, name='learning_rate')
    tvars = tf.trainable_variables()
    '''
    grads, global_norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    '''
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
  return 

def SVHN():
  conv = conv2d()
  pool = tf.nn.max_pool()
  logits = fc()
  loss = loss(logits)
  update(loss, learning_rate)  
  

# train model
def main(_):
  train_set_loader = DataLoader(TRAIN_SET_PATH, BATCH_SIZE)
  valid_set_loader = DataLoader(VALID_SET_PATH)
  
  with tf.Graph().as_default(), tf.Session() as session:
    tf.set_random_seed(time.time())
    initializer = tf.contrib.layers.xavier_initializer_conv2d(seed=)
    with tf.variable_scope("SVHN", initializer=initializer):
      train_model = SVHN()
    saver = tf.train.Saver()
    with tf.variable_scope("SVHN", reuse=True):
      valid_model = SVHN()
    
    tf.global_vairables_initializer().run()
    
    for epoch in range(MAX_EPOCHS):
      for image_batch, label_batch in train_set_loader.iter():
        session.run([], {})
        

if __name__ == "__main__":
  tf.app.run()