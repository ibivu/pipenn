# make sure comet_ml is the first import (before all other Machine learning lib)
from comet_ml import Experiment

import datetime, time, os, sys, inspect
import numpy as np

from tensorflow.keras.initializers import GlorotUniform, GlorotNormal, he_uniform, he_normal, lecun_uniform, lecun_normal, \
                                                 Orthogonal, TruncatedNormal, VarianceScaling
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Reshape, Input, concatenate, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Activation, \
                                           BatchNormalization, Conv2DTranspose, ZeroPadding2D, Dropout, UpSampling2D, UpSampling1D, \
                                           LSTM, GRU, TimeDistributed, Bidirectional, Dense, Add, ConvLSTM2D, Flatten, AveragePooling1D, \
                                           ReLU, LeakyReLU, PReLU, ELU, AlphaDropout
from PPILogger import PPILoggerCls
from PPIDataset import PPIDatasetCls, DatasetParams
from PPILoss import PPILossCls, LossParams
from PPITrainTest import PPITrainTestCls, TrainingParams, MyPReLU
from PPIParams import PPIParamsCls
from PPIParams import PPIParamsCls
from PPIExplanation import PPIExplanationCls, ExplanationParams
from tensorflow_core.python.keras.initializers import glorot_normal

class AlgParams:
    ALGRITHM_NAME = "dnet-ppi"
    DatasetParams.USE_COMET = False #True
    
    datasetLabel = 'Biolip_N'
    dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    
    ONLY_TEST = True #False
    USE_EXPLAIN = False #True
    
    USE_2D_MODEL = False
    USE_POOLING = False
    USE_BN = True
    USE_DROPOUT = True
    #DatasetParams.ONE_HOT_ENCODING = False
    #DatasetParams.USE_VAR_BATCH_INPUT = True
    
    INPUT_SHAPE = None  #must be determined in init.
    LABEL_SHAPE = None
    
    LABEL_DIM = 1
    if USE_2D_MODEL:
        PROT_IMAGE_H, PROT_IMAGE_W = 32, 32 #28, 28 #16, 16
        CNN_KERNEL_SIZE = 3
        CNN_POOL_SIZE = 4 #2
    else:
        # The receptive filed size, for a kernel-size of k and dilation rate of r can be calculated by: k+(k-1)(r-1). 
        if DatasetParams.USE_VAR_BATCH_INPUT:
            PROT_IMAGE_H,PROT_IMAGE_W = None,1
        else:
            PROT_IMAGE_H, PROT_IMAGE_W = 1170,1 #1170,1 #1024,1 #256,1 #2048,1 #576,1 #768,1 #384,1 #512,1
        
        CNN_KERNEL_SIZE = 7 #15 #4 #7
        CNN_POOL_SIZE = 2
        PROT_LEN = PROT_IMAGE_H     #PROT_LEN must be divisible by PAT_LEN.
        PAT_LEN = 8
    
    INIT_CNN_DILATION_SIZE = 1 #2 #1
    INIT_CNN_CHANNEL_SIZE = 128 #64 #32 #8 #4 #16 #128 #256
    
    if DatasetParams.USE_VAR_BATCH_INPUT:
        LossParams.USE_DELETE_PAD = True #False #True
        DatasetParams.MAX_PROT_LEN_PER_BATCH = [64,128,256,320,448,576,1170]  #False(65.13%)#True(68.57%)
        #DatasetParams.MAX_PROT_LEN_PER_BATCH = [128,512,1170] 
        #DatasetParams.MAX_PROT_LEN_PER_BATCH = [1170] 
        #DatasetParams.MAX_PROT_LEN_PER_BATCH = None
        TrainingParams.ACTIVATION_FUN = ReLU #MyPReLU(68.57%) #ReLU(68.84%)
        #TrainingParams.CUSTOM_OBJECTS = {'MyPReLU': MyPReLU}
    
    #DatasetParams.USE_MC_RESNET = True
    #DatasetParams.setMCResnetParams(PROT_LEN, PAT_LEN)
    #LossParams.setLossFun(LossParams.MC_RESNET)
    
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
        
        EX_COLUMNS2 = [
                #'normalized_hydropathy_index',
                #'3_wm_normalized_hydropathy_index','5_wm_normalized_hydropathy_index','7_wm_normalized_hydropathy_index','9_wm_normalized_hydropathy_index',
                 ]
        EX_COLUMNS = [
                     ]
        IN_COLUMNS = [
                    #'glob_stat_score',
                    ]
        
        #PPIParamsCls.setInitParams(ALGRITHM_NAME,dsParam=dsParam, dsExParam=EX_COLUMNS,dsInParam=IN_COLUMNS,dsLabelParam=dsLabelParam)   
        PPIParamsCls.setInitParams(cls.ALGRITHM_NAME,dsParam=dsParam, dsLabelParam=dsLabelParam)
        cls.setShapes()
        return 

def make2DBlock(x, channelSize,  dilationRate):
    ki = TrainingParams.KERNAL_INITIALIZER() 
    ub = TrainingParams.USE_BIAS 
    #kr = l2(TrainingParams.REG_LAMDA)
    #br = l2(TrainingParams.REG_LAMDA)
    #kr = l1_l2(TrainingParams.REG_LAMDA, TrainingParams.REG_LAMDA)
    #br = l1_l2(TrainingParams.REG_LAMDA, TrainingParams.REG_LAMDA)
    x = Conv2D(channelSize, AlgParams.CNN_KERNEL_SIZE, dilation_rate=dilationRate, padding='same', use_bias=ub, kernel_initializer=ki)(x)
    
    if AlgParams.USE_DROPOUT:
        x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    if AlgParams.USE_BN:
        x = BatchNormalization()(x)    
    
    if DatasetParams.USE_VAR_BATCH_INPUT:    
        x = TrainingParams.ACTIVATION_FUN(shared_axes=[1])(x)    #low performance
    else:
        x = TrainingParams.ACTIVATION_FUN()(x)
    
    if AlgParams.USE_POOLING:
        x = MaxPooling2D(pool_size=AlgParams.CNN_POOL_SIZE, strides=1, padding='same')(x)
    
    return x   

def make2DModel():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    channelSize = 64 
    dilationRate = AlgParams.INIT_CNN_DILATION_SIZE
    x = make2DBlock(x, channelSize, dilationRate)
    channelSize = 128
    dilationRate = dilationRate * 2
    x = make2DBlock(x, channelSize, dilationRate)
    channelSize = 128
    dilationRate = dilationRate * 2
    x = make2DBlock(x, channelSize, dilationRate)
    channelSize = 64
    dilationRate = dilationRate * 2
    x = make2DBlock(x, channelSize, dilationRate)
    channelSize = 32
    dilationRate = dilationRate * 2
    x = make2DBlock(x, channelSize, dilationRate)
    
    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
        
    model = Model(inputs=protImg, outputs=x)
    
    return model

def make2DModel2():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    channelSize = 64
    dilationRate = 1
    x = make2DBlock(x, 2, channelSize, dilationRate, True, kernalParam=3)
    x = make2DBlock(x, 2, channelSize, dilationRate, True, kernalParam=3)
    x = make2DBlock(x, 3, channelSize, dilationRate, True, kernalParam=3)
    x = make2DBlock(x, 3, channelSize, dilationRate, False, kernalParam=3)
    dilationRate = 2
    x = make2DBlock(x, 3, channelSize, dilationRate, False, kernalParam=3)
    channelSize = 4096
    dilationRate = 4
    x = make2DBlock(x, 1, channelSize, dilationRate, False, kernalParam=7)
    dilationRate = 1
    x = make2DBlock(x, 1, channelSize, dilationRate, False, kernalParam=1)
    channelSize = 1
    x = make2DBlock(x, 1, channelSize, dilationRate, False, kernalParam=3)
        
    model = Model(inputs=protImg, outputs=x)
    
    return model

def make1DBlock(x, channelSize,  dilationRate):
    ki = TrainingParams.KERNAL_INITIALIZER() 
    ub = TrainingParams.USE_BIAS 
    #kr = l2(TrainingParams.REG_LAMDA)
    #br = l2(TrainingParams.REG_LAMDA)
    #kr = l1_l2(TrainingParams.REG_LAMDA, TrainingParams.REG_LAMDA)
    #br = l1_l2(TrainingParams.REG_LAMDA, TrainingParams.REG_LAMDA)
    x = Conv1D(channelSize, AlgParams.CNN_KERNEL_SIZE, dilation_rate=dilationRate, padding='same', use_bias=ub, kernel_initializer=ki)(x)
    
    if AlgParams.USE_DROPOUT:
        x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    if AlgParams.USE_BN:
        x = BatchNormalization()(x)    
    
    if DatasetParams.USE_VAR_BATCH_INPUT:    
        x = TrainingParams.ACTIVATION_FUN(shared_axes=[1])(x)    #low performance
    else:
        x = TrainingParams.ACTIVATION_FUN()(x)
    
    if AlgParams.USE_POOLING:
        x = MaxPooling1D(pool_size=AlgParams.CNN_POOL_SIZE, strides=1, padding='same')(x)
    
    return x   

def makeDenseBlock(x, denseSize, doDropout):
    x = TimeDistributed(Dense(denseSize))(x)
    if doDropout:
        x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    x = BatchNormalization()(x)
    x = TrainingParams.ACTIVATION_FUN()(x)

    return x

def make1DModel():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg

    channelSize = 64 
    dilationRate = AlgParams.INIT_CNN_DILATION_SIZE
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 128
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 128
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 64
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 32
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    
    #x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
        
    model = Model(inputs=protImg, outputs=x)
    
    return model

def make1DModel6():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg

    channelSize = 64 
    dilationRate = AlgParams.INIT_CNN_DILATION_SIZE
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 128
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 128
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 64
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 32
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    
    #x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    #x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    
    print("====== resnet-shape: ", x.shape) #====== resnet-shape:  (?, 48, 128) 
    x = Reshape((DatasetParams.MC_RESNET_LABEL_LEN, DatasetParams.MC_RESNET_PAT_LEN * channelSize))(x)
    print("====== resnet-shape: ", x.shape) #====== resnet-shape:  (?, 8, 768)
    #for i in range()
    #x = Conv1D(filters=cs, kernel_size=ks, strides=1, padding='same', use_bias=ub, kernel_initializer=ki, kernel_regularizer=kr)(x)
    #x = MaxPooling1D(pool_size=PAT_LEN)(x)
    
    x = TimeDistributed(Dense(DatasetParams.MC_RESNET_LABEL_DIM, activation='softmax'))(x) #only matters if the layer befote is lstm.
    #x = Dense(DatasetParams.MC_RESNET_LABEL_DIM, activation='softmax')(x)
    print("====== resnet-shape: ", x.shape) #====== resnet-shape:  (?, 8, 64)
        
    model = Model(inputs=protImg, outputs=x)
    
    return model

def make1DModel5():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg

    channelSize = 64 
    dilationRate = AlgParams.INIT_CNN_DILATION_SIZE
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 128
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 128
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 64
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    channelSize = 32
    dilationRate = dilationRate * 2
    x = make1DBlock(x, channelSize, dilationRate)
    
    #x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    #x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    
    # {"sum", "mul", "ave", "concat", None}
    #x = Bidirectional(LSTM(AlgParams.RNN_CHANNEL_SIZE, return_sequences=False), merge_mode='ave')(x) 
    #x = Bidirectional(GRU(AlgParams.RNN_CHANNEL_SIZE, return_sequences=True, dropout=TrainingParams.DROPOUT_RATE), merge_mode='concat')(x) 
    #x = Bidirectional(LSTM(channelSize, return_sequences=True), merge_mode='concat')(x) 
    x = GRU(channelSize, return_sequences=True)(x)
    x = Dense(1, activation='sigmoid')(x)
        
    model = Model(inputs=protImg, outputs=x)
    
    return model

def performTraining():
    ##USE_2D_MODEL = True
    #LossParams.setLossFun(LossParams.MEAN_SQUARED)
    #no-padding
    #AlgParams.USE_DROPOUT = False
    #AlgParams.USE_BN = False
    ##AlgParams.USE_POOLING = True
    #TrainingParams.KERNAL_INITIALIZER = GlorotNormal
    #TrainingParams.ACTIVATION_FUN = ReLU
    """
    TrainingParams.KERNAL_INITIALIZER = he_uniform
    TrainingParams.ACTIVATION_FUN = PReLU
    #TrainingParams.BATCH_SIZE = 8 
    #TrainingParams.USE_EARLY_STOPPING = False    #if the training set is Serendip.
    
    TrainingParams.USE_BIAS = True
    LossParams.USE_WEIGHTED_LOSS = True
    LossParams.USE_PAD_WEIGHTS_IN_LOSS = True
    LossParams.LOSS_ONE_WEIGHT = 90.0 #0.90
    LossParams.LOSS_ZERO_WEIGHT = 10.0 #0.10
    LossParams.LOSS_PAD_WEIGHT = 0.0 #0.001
    """
    if AlgParams.USE_2D_MODEL: 
        model = make2DModel()
    else:
        model = make1DModel()
    TrainingParams.persistParams(sys.modules[__name__])
    PPITrainTestCls().trainModel(model)
    
    return

def performTesting():
    TrainingParams.GEN_METRICS_PER_PROT = True
    tstResults = PPITrainTestCls().testModel()
    return tstResults

def performExplanation():
   ExplanationParams.NUM_BKS = 5
   ExplanationParams.NUM_TSTS = 3
   ExplanationParams.MAX_FEATURES = 10 
   PPIExplanationCls.explainModel()
        
if __name__ == "__main__":
    AlgParams.initAlgParams()
    if AlgParams.USE_EXPLAIN:
        performExplanation()
        sys.exit(0)
    if not AlgParams.ONLY_TEST:
        performTraining()
    performTesting()
    
   
