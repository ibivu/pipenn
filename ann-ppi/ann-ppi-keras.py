# make sure comet_ml is the first import (before all other Machine learning lib)
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
from PPIExplanation import PPIExplanationCls, ExplanationParams, ExpTypes, GroupByTypes

class AlgParams:
    ALGRITHM_NAME = "ann-ppi"
    '''
    # For web-service use the following:
    DatasetParams.USE_COMET = False
    ONLY_TEST = True
    datasetLabel = 'UserDS_A'
    dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    '''
    DatasetParams.USE_COMET = True
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
    
    USE_EXPLAIN = False
    
    LABEL_DIM = 1
    PROT_IMAGE_H, PROT_IMAGE_W = 1, 1
    INPUT_SHAPE = None  #must be determined in init.
    LABEL_SHAPE = None
    
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

def makeDenseBlock(x, denseSize):
    x = Dense(denseSize)(x)
    x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    x = BatchNormalization()(x)    
    x = TrainingParams.ACTIVATION_FUN()(x)

    return x

def makeANNModel():
    # Relation among number of hidden layers, samples, inputs and outputs.
    # N_h = N_s / (a * (N_i + N_o)
    # 2>a<10
    HIDDEN_LAYER_1_SIZE = 1024
    HIDDEN_LAYER_2_SIZE = 768
    HIDDEN_LAYER_3_SIZE = 512
    HIDDEN_LAYER_4_SIZE = 256
    HIDDEN_LAYER_5_SIZE = 128
    HIDDEN_LAYER_6_SIZE = 64
    HIDDEN_LAYER_7_SIZE = 32
    HIDDEN_LAYER_8_SIZE = 16
    HIDDEN_LAYER_9_SIZE = 8
    HIDDEN_LAYER_10_SIZE = 4
    
    HIDDEN_LAYER_11_SIZE = 10 
    HIDDEN_LAYER_12_SIZE = 10
    HIDDEN_LAYER_13_SIZE = 10
    HIDDEN_LAYER_14_SIZE = 10
    HIDDEN_LAYER_15_SIZE = 10

    LAYERS_SIZES = [
                    HIDDEN_LAYER_1_SIZE, 
                    HIDDEN_LAYER_2_SIZE, 
                    HIDDEN_LAYER_3_SIZE, 
                    HIDDEN_LAYER_4_SIZE,
                    HIDDEN_LAYER_5_SIZE,
                    HIDDEN_LAYER_6_SIZE,
                    HIDDEN_LAYER_7_SIZE, 
                    HIDDEN_LAYER_8_SIZE, 
                    #HIDDEN_LAYER_9_SIZE, 
                    #HIDDEN_LAYER_10_SIZE,
                    #HIDDEN_LAYER_11_SIZE,
                    #HIDDEN_LAYER_12_SIZE,
                    #HIDDEN_LAYER_13_SIZE,
                    #HIDDEN_LAYER_14_SIZE,
                    #HIDDEN_LAYER_15_SIZE,
                   ] 
    
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    for layer in range(len(LAYERS_SIZES)):
        x = makeDenseBlock(x, LAYERS_SIZES[layer])
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=protImg, outputs=x)
    
    return model

def performTraining():
    LossParams.USE_WEIGHTED_LOSS = False
    DatasetParams.USE_DOWN_SAMPLING = True
    DatasetParams.DOWN_SAMPLING_THRESHOLD = 0.3
    model = makeANNModel()
    TrainingParams.persistParams(sys.modules[__name__])
    PPITrainTestCls().trainModel(model)

    return

def performTesting():
    TrainingParams.SAVE_PRED_FILE = True
    TrainingParams.GEN_METRICS_PER_PROT = True
    DatasetParams.USE_DOWN_SAMPLING = False
    tstResults = PPITrainTestCls().testModel()
    return tstResults

def performExplanation():
    #DatasetParams.USE_DOWN_SAMPLING = True
    ExplanationParams.NUM_BKS = 500 #5 #50
    ExplanationParams.NUM_TSTS = 5000 #3 #100
    ExplanationParams.EXP_TYPE = ExpTypes.KERNEL
    #ExplanationParams.EXP_TYPE = ExpTypes.PERM
    #ExplanationParams.EXP_TYPE = ExpTypes.DEEP
    ExplanationParams.GROUP_BY_TYPE = GroupByTypes.BY_WM
    #ExplanationParams.GROUP_BY_TYPE = GroupByTypes.BY_ALL
    ExplanationParams.MAX_FEATURES = 15 #53 #128 #15 
    ExplanationParams.SELECTED_FEATURE = 'domain' #'length' #'AA'#'WM_PA'
    ExplanationParams.DEPENDENT_FEATURE = None #'length'
    ExplanationParams.USE_SAVED_EXPS = True
    PPIExplanationCls.explainModel()
        
if __name__ == "__main__":
    AlgParams.initAlgParams()
    if AlgParams.USE_EXPLAIN:
        performExplanation()
        sys.exit(0)
    if not AlgParams.ONLY_TEST:
        performTraining()
    performTesting()    