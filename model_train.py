from config import setup_configs as configs
from sidekick.prepro.process import Process
from sidekick.prepro.imgtoarrayprepro import ImgtoArrPrePro
from sidekick.prepro.meanprocess import MeanProcess
from sidekick.nn.conv.resnet import ResNet
from sidekick.io.hdf5datagen import Hdf5DataGen
from sidekick.callbs.manualcheckpoint import ManualCheckpoint
from sidekick.callbs.trainmonitor import TrainMonitor
import json
import argparse
import matplotlib
# without this the training times out sometimes
matplotlib.use('Agg')
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K

# Command line arguments
ap= argparse.ArgumentParser()
ap.add_argument('-o','--output', type=str, required=True ,help="Path to output directory")
ap.add_argument('-m', '--model', help='Path to checkpointed model')
ap.add_argument('-e','--epoch', type=int, default=0, help="Starting epoch of training")
args= vars(ap.parse_args())

max_epoch=95
init_lr=1e-3
# LR decay function
def poly_lr_decay(epoch):
    base_lr= init_lr
    final_epoch= max_epoch
    # power set to 1 makes it a linear rate
    # set this to greater number to turn into a poly function
    p=1.0

    final_lr= base_lr * (1- float(epoch/final_epoch))**p
    print('Changing lr from {:.6f}->{:.6f}'.format(K.get_value(model.optimizer.lr),
                                                   final_lr))

    return final_lr

def exp_lr_decay(epoch):
    current_lr= K.get_value(model.optimizer.lr)
    new_lr= current_lr
    print('Changing lr from {:.4f}->{:.4f}'.format(current_lr,
                                                   new_lr))
    return new_lr


# Building and processing dataset
print('[NOTE]:- Building Dataset...\n')
pro= Process(64, 64)
i2a= ImgtoArrPrePro()
# Loading means from stored JSON file
data_means= json.loads(open(configs.IMG_MEAN).read())
# Using means to initialize the mean preprocessor to normalize dataset
meanpro= MeanProcess(data_means['R'], data_means['G'], data_means['B'])
# Using image augmentation to get better results
aug= ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                        shear_range=0.1, zoom_range=0.15, horizontal_flip=True, fill_mode="nearest")
# Building the HDF5 generators
train_gen= Hdf5DataGen(configs.HDF5_TRAIN, 64, 200, aug=aug, preprocessors=[pro, meanpro, i2a])
val_gen= Hdf5DataGen(configs.HDF5_VAL, 64, 200, preprocessors=[pro, meanpro, i2a])

# Compiling or loading a checkpointed model
if args['model'] is None:
    print("[NOTE]:- Building model from scratch...")
    model= ResNet.build(64, 64, 3, 200, (3,5,7), (64,128,256,512), reg_val=0.0005)
    opt= SGD(learning_rate=init_lr, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer=opt)
else:
    print("[NOTE]:- Building model {}\n".format(args['model']))
    model= load_model(args['model'])
    old_lr= K.get_value(model.optimizer.lr)
    print('[NOTE]:- Starting training from lr-> {}\n'.format(old_lr))

# Setting up callbacks
callbacks= [ManualCheckpoint(args['output'], save_at=1, start_from=args['epoch']),
            TrainMonitor(figPath=configs.FIG_PATH, jsonPath=configs.JSON_PATH, startAt=args['epoch']),
            LearningRateScheduler(poly_lr_decay)]

# Training model
print("[NOTE]:- Training model...\n")
model.fit_generator(train_gen.generator(),
                    steps_per_epoch=train_gen.data_length//64,
                    validation_data= val_gen.generator(),
                    validation_steps= val_gen.data_length//64,
                    epochs=max_epoch,
                    max_queue_size=10,
                    callbacks=callbacks,
                    initial_epoch=args['epoch'])

train_gen.close()
val_gen.close()
