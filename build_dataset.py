from config import setup_configs as conf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sidekick.io.hdf5_writer import Hdf5Writer
from imutils import paths
import cv2
import os
import json
import numpy as np
import progressbar

# get the image paths
train_paths= list(paths.list_images(conf.TRAIN_IMAGES))
train_labels= [t.split(os.path.sep)[-3] for t in train_paths]

# Initialize the Label Encoder (this gets a list of labels encoded into number)
# So since there are 200 classes, there are 200 labels
le= LabelEncoder()
train_labels= le.fit_transform(train_labels)

# Performing train-test split since there is no test set
# Notice how stratify is used to check the ratio of each label type
train_paths, test_paths, train_labels, test_labels= train_test_split(train_paths,train_labels,
                                               test_size=conf.NUM_TEST_IMAGES,
                                               stratify=train_labels)

val_file= open(conf.VAL_LABELS).read().strip().split('\n')
val_paths= [os.path.sep.join([conf.VAL_IMAGES, v.split('\t')[0]]) for v in val_file]
val_labels= [v.split('\t')[1] for v in val_file]
# Since Label Encoder has already been fitted we can transform simply
val_labels= le.transform(val_labels)

# Creating the HDF5

files= [('train', train_paths, train_labels, conf.HDF5_TRAIN),
        ('test', test_paths, test_labels, conf.HDF5_TEST),
        ('val', val_paths, val_labels, conf.HDF5_VAL)]

# To store mean of each image
R,G,B=([], [], [])
for optype, paths, labels, output_path in files:
    # Initialize the HDF5 writer with the dimensions and output path
    dat_writer= Hdf5Writer((len(paths), 64, 64, 3), output_path)

    # Initializing the progress bar display
    display=["Building Dataset: ", progressbar.Percentage(), " ",
             progressbar.Bar(), " ", progressbar.ETA()]

    # Start the progress bar
    progress= progressbar.ProgressBar(maxval=len(paths), widgets=display).start()

    # Iterate through each img path
    for (i, (p, l)) in enumerate(zip(paths,labels)):
        img= cv2.imread(p)

        # Calculating mean of each image and appending
        if optype=='train':
            b, g, r= cv2.mean(img)[:3]
            B.append(b)
            G.append(g)
            R.append(r)

        # Add data and update progress bar based on counter
        dat_writer.add([img], [l])
        progress.update(i)

    # Finish the progress for one type
    progress.finish()
    dat_writer.close()

print('\n[NOTE]:- Dumping mean to file...')
mean_dat= {'R': np.mean(R), 'G': np.mean(G), 'B': np.mean(B)}
f= open(conf.IMG_MEAN, 'w')
f.write(json.dumps(mean_dat))
f.close()





