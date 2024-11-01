import matplotlib.pyplot as plt
import datetime
import time
import os
import pandas as pd
import numpy as np

#import tensorflow as tf
import tensorflow.compat.v1 as tf

from tensorflow.python.keras.utils import plot_model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Lambda, Reshape, Input, concatenate, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Activation, \
                                           BatchNormalization, Conv2DTranspose, ZeroPadding2D, Dropout, UpSampling2D, UpSampling1D, \
                                           LSTM, GRU, TimeDistributed, Bidirectional, Dense
from PPILogger import PPILoggerCls
from PPIDataset import PPIDatasetCls, DatasetParams
from PPILoss import PPILossCls, LossParams
from PPITrainTest import PPITrainTestCls, TrainingParams

tf.enable_eager_execution()
np.random.seed(7)
tf.set_random_seed(7)

ALGRITHM_NAME = "rnn-ppi"
logger = PPILoggerCls.initLogger(ALGRITHM_NAME)

TRAINING_FILE = DatasetParams.PREPARED_COMBINED_TRAINING_FILE
#TRAINING_FILE = DatasetParams.PREPARED_HOMO_TRAINING_FILE
#TRAINING_FILE = DatasetParams.PREPARED_HETRO_TRAINING_FILE
#TRAINING_FILE = DatasetParams.BIOLIP_RANDOM_TRAINING_FILE
FEATURE_COLUMNS = [
               #'mean_H',
               'AliSeq',
               #'length', 
               ]
SEQ_COLUMN_NAME = 'sequence' 
LABEL_COLUMNS = ['p_interface']
DatasetParams.setFeatureColumns(DatasetParams.FEATURE_COLUMNS_ENH_NOWIN)
#DatasetParams.setFeatureAndLabelColumns(DatasetParams.FEATURE_COLUMNS_BIOLIP, SEQ_COLUMN_NAME, LABEL_COLUMNS)

ONLY_TEST = False

LABEL_DIM = 1
PROT_IMAGE_H, PROT_IMAGE_W = 768, 1 #1024, 1 #512, 1 #256, 1 #128, 1 #64, 1
INPUT_SHAPE = (PROT_IMAGE_H, DatasetParams.getFeaturesDim())
LABEL_SHAPE = (PROT_IMAGE_H, LABEL_DIM)

RNN_DIM = 64 #128

def makeDenseLayers(x):
    HIDDEN_LAYER_1_SIZE = 40 #32
    HIDDEN_LAYER_2_SIZE = 20 #16
    HIDDEN_LAYER_3_SIZE = 10 #8

    LAYERS_SIZES = [
                    #HIDDEN_LAYER_1_SIZE, 
                    #HIDDEN_LAYER_2_SIZE, 
                    #HIDDEN_LAYER_3_SIZE, 
                   ] 
    for layer in range(len(LAYERS_SIZES)):
        x = TimeDistributed(Dense(LAYERS_SIZES[layer]))(x)
        #x = TimeDistributed(Dropout(TrainingParams.DROPOUT_RATE))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(Activation(activation="relu"))(x)
        #x = TimeDistributed(Dropout(TrainingParams.DROPOUT_RATE))(x)
        
    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    
    return x

def makeRNNModel():
    protImg = Input(shape=INPUT_SHAPE)
    x = protImg
    
    x = LSTM(RNN_DIM, return_sequences=True)(x)
    # no improvement and progress at all. All metrics remains constant.
    #x = Bidirectional(LSTM(RNN_DIM, return_sequences=True))(x)    
    x = makeDenseLayers(x)
    
    model = Model(inputs=protImg, outputs=x)    
    
    return model
    
def setParams():
    PPIDatasetCls.setLogger(logger)
    PPILossCls.setLogger(logger)
    PPITrainTestCls.setLogger(logger)
    TrainingParams.TRAINING_FILE = TRAINING_FILE
    TrainingParams.INPUT_SHAPE = INPUT_SHAPE
    TrainingParams.LABEL_SHAPE = LABEL_SHAPE
    TrainingParams.setFileNames(ALGRITHM_NAME)
    
    #DatasetParams.TESTING_FILE_SET = [DatasetParams.BIOLIP_RANDOM_TESTING_FILE]
    DatasetParams.SKIP_SLICING = True
    
    return

def performTraining():
    setParams()
    model = makeRNNModel()
    PPITrainTestCls().trainModel(model)
    
    return

def performTesting():
    setParams()
    PPITrainTestCls().testModel()
    
    return
        
if __name__ == "__main__":
    if not ONLY_TEST:
        performTraining()
    performTesting()    