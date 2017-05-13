import scipy.io as sio
from matplotlib import pyplot as plt
import numpy as np


# configuration
BATCH_SIZE = 10
DATASET_PATH = "./data/train_32x32.mat"

class DataLoader:
  def __init__(self, data_path, batch_size):
    data = sio.loadmat(data_path)
    self.images = data['X']
    self.labels = data['y']

    image_batches = np.reshape()
    label_batches = []

  def iter(self):
    for image, label in zip(image_batches, label_batches):
      yield image, label


if __name__ == "__main__":

  dataloader = DataLoader(DATASET_PATH, BATCH_SIZE)

  batch_count = 0
  fig = plt.figure()
  for image, label in dataloader.iter():
    if batch_count > 0:
      break
    for batch_i in range(BATCH_SIZE):
      sub_plot = fig.add_subplot(1, batch_i+1, 1)
      plt.imshow(image)
    batch_count += 1

  plt.show()
