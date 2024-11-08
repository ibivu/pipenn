from comet_ml import Experiment

import datetime, time, os, sys, inspect
#import pandas as pd
import numpy as np

from tensorflow.keras.initializers import GlorotUniform, GlorotNormal, he_uniform, he_normal, lecun_uniform, lecun_normal, \
                                                 Orthogonal, TruncatedNormal, VarianceScaling
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Reshape, Input, concatenate, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Activation, \
                                           BatchNormalization, Conv2DTranspose, ZeroPadding2D, Dropout, UpSampling2D, UpSampling1D, \
                                           LSTM, GRU, TimeDistributed, Bidirectional, Dense, Add, ConvLSTM2D, Flatten, AveragePooling1D, \
                                           ReLU, LeakyReLU, PReLU, ELU
from PPILogger import PPILoggerCls
from PPIDataset import PPIDatasetCls, DatasetParams
from PPILoss import PPILossCls, LossParams
from PPITrainTest import PPITrainTestCls, TrainingParams
from PPIParams import PPIParamsCls

class AlgParams:
    ALGRITHM_NAME = "rnn-ppi"
    '''
    # For web-service use the following:
    DatasetParams.USE_COMET = False
    ONLY_TEST = True
    datasetLabel = 'UserDS_A'
    dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    '''
    DatasetParams.USE_COMET = False
    ONLY_TEST = False
    # Put this on True for PIPENN-1; it must be put on False for PIPENNEMB
    DatasetParams.ONE_HOT_ENCODING = False
    
    datasetLabel = 'Epitope'
    dataset = DatasetParams.FEATURE_COLUMNS_EPI_SEMI_NOWIN
    #datasetLabel = 'HH_Combined'
    #dataset = DatasetParams.FEATURE_COLUMNS_ENH_SEMI_WIN
    #datasetLabel = 'Cross_HH_BL_P'
    #dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    #datasetLabel = 'Cross_BL_P_HH'
    #datasetLabel = 'Cross_BL_A_HH'
    #dataset = DatasetParams.FEATURE_COLUMNS_ENH_SEMI_WIN
    #datasetLabel = 'Biolip_P'
    #dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    
    NUM_BLOCK_REPEATS = 2 #8 #3 #2
    
    LABEL_DIM = 1
    if DatasetParams.USE_VAR_BATCH_INPUT:
            PROT_IMAGE_H,PROT_IMAGE_W = None,1
    else:
        #use 768 only for Serendip en Epitope datasets
        PROT_IMAGE_H, PROT_IMAGE_W = 1024,1 #2048,1 #576, 1 #768, 1 #384, 1 #512, 1
    INPUT_SHAPE = None  #must be determined in init.
    LABEL_SHAPE = None
    
    RNN_DIM = 128 #64 #128
    
    @classmethod
    def setShapes(cls):
        cls.INPUT_SHAPE = (cls.PROT_IMAGE_H, DatasetParams.getFeaturesDim())   
        cls.LABEL_SHAPE = (cls.PROT_IMAGE_H, cls.LABEL_DIM)
        PPIParamsCls.setShapeParams(cls.INPUT_SHAPE, cls.LABEL_SHAPE)   
        return
    
    @classmethod
    def initAlgParams(cls, dsParam=dataset, dsLabelParam=datasetLabel, ensembleTesting=False):
        pipennDatasetLabel = PPIParamsCls.setPipennParams()
        if pipennDatasetLabel != None:
            dsLabelParam = pipennDatasetLabel
            cls.datasetLabel = pipennDatasetLabel
        
        if ensembleTesting:
            cls.ONLY_TEST = True
        else:
            PPIParamsCls.setLoggers(cls.ALGRITHM_NAME, dsLabelParam)
            
        PPIParamsCls.setInitParams(cls.ALGRITHM_NAME,dsParam=dsParam, dsLabelParam=dsLabelParam)
        cls.setShapes()
        return 

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
        x = TimeDistributed(Dropout(TrainingParams.DROPOUT_RATE))(x)
        x = TimeDistributed(BatchNormalization())(x)
        #x = TimeDistributed(Activation(activation="relu"))(x)
        x = TimeDistributed(TrainingParams.ACTIVATION_FUN())(x)
        #x = TimeDistributed(Dropout(TrainingParams.DROPOUT_RATE))(x)
    #print(x.shape)    
    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    
    return x

def makeRNNModel():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    #ki = TrainingParams.KERNAL_INITIALIZER() 
    #ub = TrainingParams.USE_BIAS 
    
    # no improvement and progress at all. All metrics remains constant.
    #x = Bidirectional(GRU(RNN_DIM, return_sequences=True))(x)  
    
    # no improvement and very long convergence  
    #x = GRU(RNN_DIM, activation=TrainingParams.ACTIVATION_FUN(), use_bias=ub, kernel_initializer=ki, return_sequences=True)(x)
    #x = GRU(RNN_DIM, recurrent_dropout=TrainingParams.DROPOUT_RATE, use_bias=ub, kernel_initializer=ki, return_sequences=True)(x)  
    #x = GRU(RNN_DIM, use_bias=ub, kernel_initializer=ki, return_sequences=True)(x) 
    #x = GRU(RNN_DIM, recurrent_dropout=TrainingParams.DROPOUT_RATE, return_sequences=True)(x) 
    for i in range(AlgParams.NUM_BLOCK_REPEATS):
        x = GRU(AlgParams.RNN_DIM, return_sequences=True)(x)
    x = makeDenseLayers(x)
    
    model = Model(inputs=protImg, outputs=x)    
    
    return model

def make1DBlock(x, rnnDimSize):
    x = GRU(rnnDimSize, return_sequences=True)(x)
    #x = Dropout(TrainingParams.DROPOUT_RATE)(x)    #very bad performance
    #x = BatchNormalization()(x)    
    #x = TrainingParams.ACTIVATION_FUN()(x)
    
    return x 

def makeRNNModel2():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg

    rnnDimSize = 64 
    x = make1DBlock(x, rnnDimSize) 
    rnnDimSize = 128
    x = make1DBlock(x, rnnDimSize) 
    rnnDimSize = 128
    x = make1DBlock(x, rnnDimSize) 
    rnnDimSize = 64
    x = make1DBlock(x, rnnDimSize) 
    rnnDimSize = 32
    x = make1DBlock(x, rnnDimSize) 
    
    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    
    model = Model(inputs=protImg, outputs=x)    
    
    return model

def performTraining():
    LossParams.USE_DELETE_PAD = False #False(71.32%,65.79%) #True(69.08%,62.04%)
    model = makeRNNModel()
    TrainingParams.persistParams(sys.modules[__name__])
    PPITrainTestCls().trainModel(model)

    return

def performTesting():
    TrainingParams.SAVE_PRED_FILE = True
    TrainingParams.GEN_METRICS_PER_PROT = True
    tstResults = PPITrainTestCls().testModel()
    return tstResults
        
if __name__ == "__main__":
    AlgParams.initAlgParams()
    if not AlgParams.ONLY_TEST:
        performTraining()
    performTesting()      