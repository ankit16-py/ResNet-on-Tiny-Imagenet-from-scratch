from config import setup_configs as config
from sidekick.prepro.process import Process
from sidekick.prepro.meanprocess import MeanProcess
from sidekick.prepro.imgtoarrayprepro import ImgtoArrPrePro
from tensorflow.keras.models import load_model
from sidekick.eval.calc_ranked import calculate_ranked
from sidekick.io.hdf5datagen import Hdf5DataGen
import json
import argparse

# Building commandline arguments
ap= argparse.ArgumentParser()
ap.add_argument('-m','--model',required=True, help="Path to trained model")
args= vars(ap.parse_args())

# Building the testing dataset
print('[NOTE]:- Loading testing data...\n')
data_means= json.loads(open(config.IMG_MEAN).read())
pro= Process(64,64)
meanpro= MeanProcess(data_means['R'], data_means['G'], data_means['B'])
i2a= ImgtoArrPrePro()
test_gen= Hdf5DataGen(config.HDF5_TEST, 64, 200, preprocessors=[pro, meanpro, i2a])

# Building model
print('[NOTE]:- Building model and evaluating...\n')
model= load_model(args['model'])
preds= model.predict_generator(test_gen.generator(),
                               steps= test_gen.data_length//64,
                               max_queue_size=10)
# calculating rank accuracies
rank1, rank5= calculate_ranked(preds, test_gen.db['Labels'])
print("Rank-1:- {:.2f}% & Rank-5:- {:.2f}%".format(rank1*100, rank5*100))
test_gen.close()