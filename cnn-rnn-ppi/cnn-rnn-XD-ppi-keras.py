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
    ALGRITHM_NAME = "cnn-rnn-ppi"
    DatasetParams.USE_COMET = False #True
    
    datasetLabel = 'Biolip_N'
    dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    
    ONLY_TEST = True #False
    USE_2D_MODEL = False
    NUM_BLOCK_REPEATS = 8 #8(73.44%,61.57%) #19(72.24%,61.54%)
    RNN_DIM = 128
    
    INPUT_SHAPE = None  #must be determined in init.
    LABEL_SHAPE = None
    
    LABEL_DIM = 1
    if USE_2D_MODEL:
        PROT_IMAGE_H, PROT_IMAGE_W = 32,32 #24, 24 #576, 576 #8, 8 #16, 16
        CNN_CHANNEL_SIZE = 64 #64 > 128
        RNN_CHANNEL_SIZE = 64 #64 > 128
        RNN_SEQ_SIZE = int((PROT_IMAGE_H * PROT_IMAGE_W * CNN_CHANNEL_SIZE) // RNN_CHANNEL_SIZE)
        RNN_INPUT_SHAPE = (RNN_SEQ_SIZE, RNN_CHANNEL_SIZE)
    else:
        if DatasetParams.USE_VAR_BATCH_INPUT:
            PROT_IMAGE_H,PROT_IMAGE_W = None,1
        else:
            PROT_IMAGE_H, PROT_IMAGE_W = 1024,1 #1170,1 #1024,1 #256,1 #2048,1 #576,1 #768,1 #384,1 #512,1
        CNN_CHANNEL_SIZE = 128 #60 #256 #128 > 64
        RNN_CHANNEL_SIZE = 200 #128 #64
        RNN_SEQ_SIZE = int((PROT_IMAGE_H * PROT_IMAGE_W * CNN_CHANNEL_SIZE) // RNN_CHANNEL_SIZE)
        RNN_INPUT_SHAPE = (RNN_SEQ_SIZE, RNN_CHANNEL_SIZE)
        
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
        if ensembleTesting:
            cls.ONLY_TEST = True
        else:
            PPIParamsCls.setLoggers(cls.ALGRITHM_NAME, dsLabelParam)

        PPIParamsCls.setInitParams(cls.ALGRITHM_NAME,dsParam=dsParam, dsLabelParam=dsLabelParam)
        cls.setShapes()
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

def make2DRNNBlock(x):
    res = x
    #print(x.shape)
    x = make2DConv(x, 1)
    x = makeActNormBlock(x)
    x = Flatten()(x)
    #print(x.shape)
    x = Reshape(AlgParams.RNN_INPUT_SHAPE)(x)
    #print(x.shape)
    
    # Bidir > GRU; LSTM > GRU; ave > mul;
    # {"sum", "mul", "ave", "concat", None}
    x = Bidirectional(LSTM(AlgParams.RNN_CHANNEL_SIZE, return_sequences=False), merge_mode='ave')(x) 
    #x = Bidirectional(GRU(AlgParams.RNN_CHANNEL_SIZE, return_sequences=False), merge_mode='ave')(x) 
    #x = GRU(AlgParams.RNN_CHANNEL_SIZE, return_sequences=False)(x)
    
    #print(x.shape)
    x = Add()([res,x])
    #print(x.shape)
    
    return x
    
def make2DModel():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    x = make2DConv(x, 3)
    for i in range(19):
        x = make2DResBlock(x)
    
    x = makeActNormBlock(x)    
    
    for i in range(2):
        x = make2DRNNBlock(x)
    
    #x = Dense(256)(x)
    x = Dense(128)(x)
    x = TrainingParams.ACTIVATION_FUN()(x)
    x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=protImg, outputs=x)
    
    return model  

def make1DConv2(x, ks, cs=AlgParams.CNN_CHANNEL_SIZE):
    ki = TrainingParams.KERNAL_INITIALIZER() 
    ub = TrainingParams.USE_BIAS 
    
    x = Conv1D(filters=cs, kernel_size=ks, strides=1, padding='same', use_bias=ub, kernel_initializer=ki)(x)
    
    return x

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

def make1DRNNBlock(x):
    x = GRU(AlgParams.RNN_DIM, return_sequences=True)(x)
    x = GRU(AlgParams.RNN_DIM, return_sequences=True)(x)
    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)

    return x
    
def make1DRNNBlock2(x):
    #res = x
    #print(x.shape)
    ##x = make1DConv(x, 1, 2*AlgParams.RNN_CHANNEL_SIZE)
    x = make1DConv(x, 1)
    #x = make1DConv(x, 1, AlgParams.RNN_CHANNEL_SIZE)
    x = makeActNormBlock(x)
    res = x
    #print(x.shape)
    #x = Flatten()(x)
    #print(x.shape)
    #x = Reshape(AlgParams.RNN_INPUT_SHAPE)(x)
    #print(x.shape)
    # {"sum", "mul", "ave", "concat", None}
    #x = Bidirectional(LSTM(AlgParams.RNN_CHANNEL_SIZE, return_sequences=False), merge_mode='ave')(x) 
    #x = Bidirectional(GRU(AlgParams.RNN_CHANNEL_SIZE, return_sequences=True, dropout=TrainingParams.DROPOUT_RATE), merge_mode='concat')(x) 
    ##x = Bidirectional(LSTM(AlgParams.RNN_CHANNEL_SIZE, return_sequences=True), merge_mode='concat')(x) 
    #x = GRU(AlgParams.RNN_CHANNEL_SIZE, return_sequences=True)(x)
    x = GRU(AlgParams.RNN_DIM, return_sequences=True)(x)
    x = GRU(AlgParams.RNN_DIM, return_sequences=True)(x) 
    #print(x.shape)
    #x = make1DConv(x, 1)
    #x = makeActNormBlock(x)
    x = Add()([res,x])
    #print(x.shape)
    
    return x    

def make1DModel():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    x = make1DConv(x, 3)
    for i in range(AlgParams.NUM_BLOCK_REPEATS):
        x = make1DResBlock(x)
    
    x = makeActNormBlock(x)    
    x = make1DRNNBlock(x)
        
    model = Model(inputs=protImg, outputs=x)
    
    return model
    
def make1DModel2():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    x = make1DConv(x, 3)
    for i in range(AlgParams.NUM_BLOCK_REPEATS):
        x = make1DResBlock(x)
    
    x = makeActNormBlock(x)    
    for i in range(2):
        #x = make1DRNNBlock(x)
        x = make1DRNNBlock2(x)
    
    #x = Dense(256)(x)
    x = Dense(400)(x)
    #print(x.shape)
    #x = Activation('elu')(x)
    x = TrainingParams.ACTIVATION_FUN()(x)
    x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    x = Dense(1, activation='sigmoid')(x)
        
    model = Model(inputs=protImg, outputs=x)
    
    return model

def performTraining():
    if AlgParams.USE_2D_MODEL:
        model = make2DModel()
    else:
        model = make1DModel()
        #model = make1DModel2()    #performs lower
    TrainingParams.persistParams(sys.modules[__name__])    
    PPITrainTestCls().trainModel(model)
    
    return

def performTesting():
    TrainingParams.GEN_METRICS_PER_PROT = True
    tstResults = PPITrainTestCls().testModel()
    return tstResults
        
if __name__ == "__main__":
    AlgParams.initAlgParams()
    if not AlgParams.ONLY_TEST:
        performTraining()
    performTesting()
    
    
   
