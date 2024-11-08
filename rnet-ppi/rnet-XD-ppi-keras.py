from comet_ml import Experiment

import datetime, time, os, sys, inspect
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
    ALGRITHM_NAME = "rnet-ppi"
    '''
    # For web-service use the following:
    DatasetParams.USE_COMET = False
    ONLY_TEST = True
    datasetLabel = 'UserDS_A'
    dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    '''
    DatasetParams.USE_COMET = True
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
    
    USE_2D_MODEL = False
    NUM_BLOCK_REPEATS = 8 #8(73.44%,61.57%) #19(72.24%,61.54%)
    
    INPUT_SHAPE = None  #must be determined in init.
    LABEL_SHAPE = None
    
    LABEL_DIM = 1
    if USE_2D_MODEL:
        PROT_IMAGE_H, PROT_IMAGE_W = 32,32 #24, 24 #576, 576 #8, 8 #16, 16
        CNN_CHANNEL_SIZE = 64 #64 > 128
    else:
        if DatasetParams.USE_VAR_BATCH_INPUT:
            PROT_IMAGE_H,PROT_IMAGE_W = None,1
        else:
            PROT_IMAGE_H, PROT_IMAGE_W = 1024,1 #1170,1 #1024,1 #256,1 #2048,1 #576,1 #768,1 #384,1 #512,1
        CNN_CHANNEL_SIZE = 128 #256 #128 > 64
        
    @classmethod
    def setTestingDatasets(cls):
        DatasetParams.EXPR_TESTING_FILE_SET = [DatasetParams.PREPARED_ZK448_WIN_P_BENCHMARK_FILE, DatasetParams.PREPARED_BIOLIP_WIN_P_TESTING_FILE]
        return 
    
    @classmethod
    def setShapes(cls):
        if cls.USE_2D_MODEL:
            cls.INPUT_SHAPE = (cls.PROT_IMAGE_H, cls.PROT_IMAGE_W, DatasetParams.getFeaturesDim())
            cls.LABEL_SHAPE = (cls.PROT_IMAGE_H, cls.PROT_IMAGE_W, LABEL_DIM)
        else:
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
        #cls.setTestingDatasets()
        return 

def makeActNormBlock(x):
    x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    x = BatchNormalization()(x)    
    x = TrainingParams.ACTIVATION_FUN()(x)
    
    return x

def make2DConv(x, ks):
    ki = TrainingParams.KERNAL_INITIALIZER() 
    ub = TrainingParams.USE_BIAS 
    cs = AlgParams.CNN_CHANNEL_SIZE
    
    x = Conv2D(filters=cs, kernel_size=ks, strides=1, padding='same', use_bias=ub, kernel_initializer=ki)(x)
    
    return x

def make2DResBlock(x):
    res = x
    x = makeActNormBlock(x)
    x = make2DConv(x, 5)
    x = makeActNormBlock(x)
    x = make2DConv(x, 3)
    x = Add()([res,x])
    
    return x
    
def make2DModel():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    x = make2DConv(x, 3)
    for i in range(AlgParams.NUM_BLOCK_REPEATS):
        x = make2DResBlock(x)
    
    x = makeActNormBlock(x)    
    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    
    model = Model(inputs=protImg, outputs=x)
    
    return model  

def make1DConv(x, ks):
    ki = TrainingParams.KERNAL_INITIALIZER() 
    ub = TrainingParams.USE_BIAS 
    cs = AlgParams.CNN_CHANNEL_SIZE
    
    x = Conv1D(filters=cs, kernel_size=ks, strides=1, padding='same', use_bias=ub, kernel_initializer=ki)(x)
    
    return x

def make1DResBlock(x):
    res = x
    x = makeActNormBlock(x)
    x = make1DConv(x, 5)
    x = makeActNormBlock(x)
    x = make1DConv(x, 3)
    x = Add()([res,x])
    
    return x  

def make1DModel():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    x = make1DConv(x, 3)
    for i in range(AlgParams.NUM_BLOCK_REPEATS):
        x = make1DResBlock(x)
    
    x = makeActNormBlock(x)    
    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
        
    model = Model(inputs=protImg, outputs=x)
    
    return model

def performTraining():
    if AlgParams.USE_2D_MODEL:
        model = make2DModel()
    else:
        model = make1DModel()
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
