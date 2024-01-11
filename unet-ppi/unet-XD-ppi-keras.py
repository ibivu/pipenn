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
    ALGRITHM_NAME = "unet-ppi"
    '''
    # For web-service use the following:
    DatasetParams.USE_COMET = False
    ONLY_TEST = True
    datasetLabel = 'UserDS_A'
    dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    '''
    DatasetParams.USE_COMET = False
    ONLY_TEST = False
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
        PROT_IMAGE_H, PROT_IMAGE_W = 32,32 
        UNET_KERNEL_SIZE = 3
    else:
        if DatasetParams.USE_VAR_BATCH_INPUT:
            PROT_IMAGE_H,PROT_IMAGE_W = None,1
        else:
            PROT_IMAGE_H, PROT_IMAGE_W = 1024,1 #1170,1 #1024,1 #256,1 #2048,1 #576,1 #768,1 #384,1 #512,1
        UNET_KERNEL_SIZE = 7
    INIT_UNET_CHANNEL_SIZE = 64 #32 #8 #4 #16 #128 #256
    UNET_LEVEL_SIZE = 3 #2 #1 #3
   
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
        return 

def make2DBlock(x, isConv, channelSize):
    ki = TrainingParams.KERNAL_INITIALIZER() 
    ub = TrainingParams.USE_BIAS 
    if isConv:
        x = Conv2D(channelSize, AlgParams.UNET_KERNEL_SIZE, padding='same', use_bias=ub, kernel_initializer=ki)(x)
    else:
        x = Conv2DTranspose(channelSize, AlgParams.UNET_KERNEL_SIZE, strides=2, padding='same', use_bias=ub, kernel_initializer=ki)(x)
    x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    x = BatchNormalization()(x)    
    x = TrainingParams.ACTIVATION_FUN()(x)
    
    return x

def make2DModel():
    # Based on TernausNet model
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    channelSize = AlgParams.INIT_UNET_CHANNEL_SIZE // 2
    copyLayers = []
    levelSize = 2
    for level in range(0, levelSize, 1):
        channelSize = channelSize * 2
        x = make2DBlock(x, True, channelSize)
        copyLayers.append(x)
        x = MaxPooling2D(pool_size=2)(x)
    
    levelSize = 2
    for level in range(0, levelSize, 1):
        channelSize = channelSize * 2
        x = make2DBlock(x, True, channelSize)
        x = make2DBlock(x, True, channelSize)
        copyLayers.append(x)
        x = MaxPooling2D(pool_size=2)(x)
    
    levelSize = 1 #0 #1
    for level in range(0, levelSize, 1):
        x = make2DBlock(x, True, channelSize)
        x = make2DBlock(x, True, channelSize)
        copyLayers.append(x)
        x = MaxPooling2D(pool_size=2)(x)
    
    x = make2DBlock(x, True, channelSize)
    
    levelSize = 4 #3 #4
    channelSize = channelSize // 2
    for level in range(levelSize, 2, -1):    
        x = make2DBlock(x, False, channelSize)
        x = concatenate([copyLayers[level], x], axis=3)
        x = make2DBlock(x, True, channelSize * 2)
        
    levelSize = 2
    for level in range(levelSize, -1, -1):
        channelSize = channelSize // 2
        x = make2DBlock(x, False, channelSize)
        x = concatenate([copyLayers[level], x], axis=3)
        if level != 0:
            x = make2DBlock(x, True, channelSize * 2)    
    
    channelSize = 1
    x = Conv2D(channelSize, AlgParams.UNET_KERNEL_SIZE, padding='same')(x)
    x = Activation('sigmoid')(x)
        
    model = Model(inputs=protImg, outputs=x)
    
    return model

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same'):
    ki = TrainingParams.KERNAL_INITIALIZER() 
    ub = TrainingParams.USE_BIAS 
    # Tensorflow gives: raise NotImplementedError
    #print(input_tensor.shape)
    n,h,c = input_tensor.shape
    #x = Lambda(lambda x: tf.expand_dims(x, axis=2))(input_tensor)
    x = Reshape((h,1,c))(input_tensor)
    #print(x.shape)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding, use_bias=ub, kernel_initializer=ki)(x)
    #print(x.shape)
    #x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    n,h,w,c = x.shape
    x = Reshape((h,c))(x)
    #print(x.shape)
    
    return x

def make1DBlock(x, isConv, channelSize):
    ki = TrainingParams.KERNAL_INITIALIZER() 
    ub = TrainingParams.USE_BIAS 
    if isConv:
        x = Conv1D(channelSize, AlgParams.UNET_KERNEL_SIZE, padding='same', use_bias=ub, kernel_initializer=ki)(x)
    else:
        x = Conv1DTranspose(input_tensor=x, filters=channelSize, kernel_size=AlgParams.UNET_KERNEL_SIZE)
        #x = UpSampling1D(size=2)(x) 
    x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    x = BatchNormalization()(x)    
    x = TrainingParams.ACTIVATION_FUN()(x)
    
    return x

def make1DModel():
    # Based on TernausNet model
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    channelSize = AlgParams.INIT_UNET_CHANNEL_SIZE // 2
    copyLayers = []
    levelSize = 2
    for level in range(0, levelSize, 1):
        channelSize = channelSize * 2
        x = make1DBlock(x, True, channelSize)
        copyLayers.append(x)
        x = MaxPooling1D(pool_size=2)(x)
    
    levelSize = 2
    for level in range(0, levelSize, 1):
        channelSize = channelSize * 2
        x = make1DBlock(x, True, channelSize)
        x = make1DBlock(x, True, channelSize)
        copyLayers.append(x)
        x = MaxPooling1D(pool_size=2)(x)
    
    x = make1DBlock(x, True, channelSize)
    x = make1DBlock(x, True, channelSize)
    copyLayers.append(x)
    x = MaxPooling1D(pool_size=2)(x)
    
    x = make1DBlock(x, True, channelSize)
    
    levelSize = 4
    channelSize = channelSize // 2
    for level in range(levelSize, 2, -1):
        x = make1DBlock(x, False, channelSize)
        x = concatenate([copyLayers[level], x], axis=2)
        x = make1DBlock(x, True, channelSize * 2)
        
    levelSize = 2
    for level in range(levelSize, -1, -1):
        channelSize = channelSize // 2
        x = make1DBlock(x, False, channelSize)
        x = concatenate([copyLayers[level], x], axis=2)
        if level != 0:
            x = make1DBlock(x, True, channelSize * 2)    
    
    channelSize = 1
    x = Conv1D(channelSize, AlgParams.UNET_KERNEL_SIZE, padding='same')(x)
    x = Activation('sigmoid')(x)
        
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
    
