import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np
import os


# configuration
BATCH_SIZE = 10
DATASET_PATH = "./data/train_32x32.mat"

class DataLoader:
  def __init__(self, data_path, batch_size):
    self.path = data_path
    self.batch_size = batch_size

    sio.loadmat(path)

  def load_data():


  def iter(self):
    for image, label in zip():
      yield image, label


if __name__ == "__main__":

  dataloader = DataLoader(DATASET_PATH, BATCH_SIZE)
  dataloader.load_data()

  batch_count = 0
  fig = plt.figure()
  for image, label in dataloader.iter():
    if batch_count > 0:
      break
    for batch_i in range(BATCH_SIZE):
      sub_plot = fig.add_subplot(,,)
      plt.imshow(image[])
    batch_count += 1

  plt.show()
