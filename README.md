# AI_Ass_SVHN
The solution to the assignment of Artificial Intelligence course.

# Dataset
We adopted the Format 2 of [The Street View House Numbers (SVHN) Dataset](http://ufldl.stanford.edu/housenumbers/).

# Requirements
Ubuntu 14.04.5 LTS

python 2.7.6

tensorflow 1.0.1

# Usage
## train
python train.py
## evaluation
python test.py

# Miscellaneous
## directory tree
work_directory
| 1155081867
  | dataloader.py
  | model.py
  | test.py
  | trained_model
    | checkpoint
    | events.out.tfevents.1495168215.ip-172-31-4-31
    | graph.pbtxt
    | model.ckpt-17001.data-00000-of-00001
    | model.ckpt-17001.index
    | model.ckpt-17001.meta
  | train.py
| data
  | train.mat
  | test_images.mat
## trained model
trained model is saved in "./trained_model".
## test output
"./labels.txt" would be generated after running "python test.py".
## data files
"train.mat" <- "train_32x32.mat",
"test_images.mat" <- randomly generated from "test_32x32.mat".
all the two files are supposed be supplied by you.
# Precision
test precision is supposed to be over 90%.
