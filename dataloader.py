from __future__ import print_function
import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import math


# configuration
BATCH_SIZE = 50
NUM_SUBPLOT_COLS = 10
DATASET_PATH = "./data/train_32x32.mat"

class DataLoader:
  def __init__(self, data_path, batch_size):
    data = sio.loadmat(data_path)
    self.images = data['X']
    self.labels = data['y']

    num_samples = self.labels.shape[0]
    num_batches = num_samples // batch_size
    # last_batch_size = num_samples - num_batches * batch_size
    split_conf = [batch_size*(i+1) for i in range(num_batches)]
    split_conf.append(num_samples)

    self.image_batches = np.split(np.transpose(self.images, (3, 0, 1, 2)), split_conf, axis=0)
    self.label_batches = np.split(self.labels, split_conf, axis=0)

  def iter(self):
    for image, label in zip(self.image_batches, self.label_batches):
      yield image, label


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
