import os

# Defining paths to training and validation images
# REMEMBER PATHS HAVE TO DEFINED BASED ON WHICH SCRIPT
# THEY WILL BE UPLOADED FROM
TRAIN_IMAGES= "..\\..\\datasets\\tiny-imagenet-200\\train\\"
VAL_IMAGES= "..\\..\\datasets\\tiny-imagenet-200\\val\\images\\"

# WordNet and validation files used to generate class labels
WORDN_IDS= "..\\..\\datasets\\tiny-imagenet-200\\wnids.txt"
WORDN_LABELS= "..\\..\\datasets\\tiny-imagenet-200\\words.txt"
VAL_LABELS= "..\\..\\datasets\\tiny-imagenet-200\\val\\val_annotations.txt"

NUM_CLASSES= 200
NUM_TEST_IMAGES= 30*NUM_CLASSES

# Paths to the saved hdf5 dataset
HDF5_TRAIN= "..\\..\\datasets\\tiny-imagenet-200\\hdf5\\train.hdf5"
HDF5_VAL= "..\\..\\datasets\\tiny-imagenet-200\\hdf5\\val.hdf5"
HDF5_TEST= "..\\..\\datasets\\tiny-imagenet-200\\hdf5\\test.hdf5"

# Mean stored to compute mean-normalization on images.
IMG_MEAN= "tiny_imagenet_mean.json"

# Output file paths
OUTPUT= "outs"
JSON_PATH= os.path.sep.join([OUTPUT, "training_metrics.json"])
FIG_PATH= os.path.sep.join([OUTPUT, "training_plot.jpg"])