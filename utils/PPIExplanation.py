import datetime, time, random
import enum
import joblib
from itertools import repeat, chain
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Flatten

import shap

from PPIDataset import PPIDatasetCls, DatasetParams
from PPITrainTest import TrainingParams

logger = None

class ExpTypes(enum.Enum):
    PERM = 0
    KERNEL = 1
    DEEP = 2

class GroupByTypes(enum.Enum):
    BY_ALL = 0
    BY_WM = 1

class ExplanationParams(object):
    NUM_BKS = 5 #10 #50 #500 #100 #50                  #number of backgrounds
    NUM_TSTS = 3 #5 #3000 #5                           #number of test examples (AAs)
    MAX_FEATURES = 10
    AA_START_INDX = 0                                  #index of one instance in the test examples
    INST_IND = 0
    MODEL_EXP_FILE = "../models/{}/EXP-{}.exp"
    USE_SAVED_EXPS = False
    EXP_TYPE = ExpTypes.KERNEL
    GROUP_BY_TYPE = GroupByTypes.BY_ALL
    SELECTED_FEATURE = None
    DEPENDENT_FEATURE = None
    
    """
    BIOLIP_FEATURE_NAMES = [
                'domain',
                'length',
                'ASA_q',
                'RSA_q',
                'PB_q',
                'PA_q',
                'PC_q',
                'pssm_A','pssm_R','pssm_N','pssm_D','pssm_C','pssm_Q','pssm_E','pssm_G','pssm_H','pssm_I',
                'pssm_L','pssm_K','pssm_M','pssm_F','pssm_P','pssm_S','pssm_T','pssm_W','pssm_Y','pssm_V',
                '3_wm_ASA_q','5_wm_ASA_q','7_wm_ASA_q','9_wm_ASA_q',
                '3_wm_RSA_q','5_wm_RSA_q','7_wm_RSA_q','9_wm_RSA_q',
                '3_wm_PB_q','5_wm_PB_q','7_wm_PB_q','9_wm_PB_q',
                '3_wm_PA_q','5_wm_PA_q','7_wm_PA_q','9_wm_PA_q',
                '3_wm_PC_q','5_wm_PC_q','7_wm_PC_q','9_wm_PC_q',
                '3_wm_pssm_A','3_wm_pssm_R','3_wm_pssm_N','3_wm_pssm_D','3_wm_pssm_C','3_wm_pssm_Q','3_wm_pssm_E',
                '3_wm_pssm_G','3_wm_pssm_H','3_wm_pssm_I','3_wm_pssm_L','3_wm_pssm_K','3_wm_pssm_M','3_wm_pssm_F',
                '3_wm_pssm_P','3_wm_pssm_S','3_wm_pssm_T','3_wm_pssm_W','3_wm_pssm_Y','3_wm_pssm_V',
                '5_wm_pssm_A','5_wm_pssm_R','5_wm_pssm_N','5_wm_pssm_D','5_wm_pssm_C','5_wm_pssm_Q','5_wm_pssm_E',
                '5_wm_pssm_G','5_wm_pssm_H','5_wm_pssm_I','5_wm_pssm_L','5_wm_pssm_K','5_wm_pssm_M','5_wm_pssm_F',
                '5_wm_pssm_P','5_wm_pssm_S','5_wm_pssm_T','5_wm_pssm_W','5_wm_pssm_Y','5_wm_pssm_V',
                '7_wm_pssm_A','7_wm_pssm_R','7_wm_pssm_N','7_wm_pssm_D','7_wm_pssm_C','7_wm_pssm_Q','7_wm_pssm_E',
                '7_wm_pssm_G','7_wm_pssm_H','7_wm_pssm_I','7_wm_pssm_L','7_wm_pssm_K','7_wm_pssm_M','7_wm_pssm_F',
                '7_wm_pssm_P','7_wm_pssm_S','7_wm_pssm_T','7_wm_pssm_W','7_wm_pssm_Y','7_wm_pssm_V',
                '9_wm_pssm_A','9_wm_pssm_R','9_wm_pssm_N','9_wm_pssm_D','9_wm_pssm_C','9_wm_pssm_Q','9_wm_pssm_E',
                '9_wm_pssm_G','9_wm_pssm_H','9_wm_pssm_I','9_wm_pssm_L','9_wm_pssm_K','9_wm_pssm_M','9_wm_pssm_F',
                '9_wm_pssm_P','9_wm_pssm_S','9_wm_pssm_T','9_wm_pssm_W','9_wm_pssm_Y','9_wm_pssm_V',
                'AA',
                ]
    
    {'domain': 0, 'length': 1, 'ASA_q': 2, 'RSA_q': 3, 'PB_q': 4, 'PA_q': 5, 'PC_q': 6, 
    'pssm_A': 7, 'pssm_R': 8, 'pssm_N': 9, 'pssm_D': 10, 'pssm_C': 11, 'pssm_Q': 12, 'pssm_E': 13, 'pssm_G': 14, 
    'pssm_H': 15, 'pssm_I': 16, 'pssm_L': 17, 'pssm_K': 18, 'pssm_M': 19, 'pssm_F': 20, 'pssm_P': 21, 'pssm_S': 22, 
    'pssm_T': 23, 'pssm_W': 24, 'pssm_Y': 25, 'pssm_V': 26, 
    '3_wm_ASA_q': 27, '5_wm_ASA_q': 28, '7_wm_ASA_q': 29, '9_wm_ASA_q': 30, '3_wm_RSA_q': 31, '5_wm_RSA_q': 32, '7_wm_RSA_q': 33, 
    '9_wm_RSA_q': 34, '3_wm_PB_q': 35, '5_wm_PB_q': 36, '7_wm_PB_q': 37, '9_wm_PB_q': 38, '3_wm_PA_q': 39, '5_wm_PA_q': 40, 
    '7_wm_PA_q': 41, '9_wm_PA_q': 42, '3_wm_PC_q': 43, '5_wm_PC_q': 44, '7_wm_PC_q': 45, '9_wm_PC_q': 46, '3_wm_pssm_A': 47, 
    '3_wm_pssm_R': 48, '3_wm_pssm_N': 49, '3_wm_pssm_D': 50, '3_wm_pssm_C': 51, '3_wm_pssm_Q': 52, '3_wm_pssm_E': 53, 
    '3_wm_pssm_G': 54, '3_wm_pssm_H': 55, '3_wm_pssm_I': 56, '3_wm_pssm_L': 57, '3_wm_pssm_K': 58, '3_wm_pssm_M': 59, 
    '3_wm_pssm_F': 60, '3_wm_pssm_P': 61, '3_wm_pssm_S': 62, '3_wm_pssm_T': 63, '3_wm_pssm_W': 64, '3_wm_pssm_Y': 65, 
    '3_wm_pssm_V': 66, '5_wm_pssm_A': 67, '5_wm_pssm_R': 68, '5_wm_pssm_N': 69, '5_wm_pssm_D': 70, '5_wm_pssm_C': 71, 
    '5_wm_pssm_Q': 72, '5_wm_pssm_E': 73, '5_wm_pssm_G': 74, '5_wm_pssm_H': 75, '5_wm_pssm_I': 76, '5_wm_pssm_L': 77, 
    '5_wm_pssm_K': 78, '5_wm_pssm_M': 79, '5_wm_pssm_F': 80, '5_wm_pssm_P': 81, '5_wm_pssm_S': 82, '5_wm_pssm_T': 83, 
    '5_wm_pssm_W': 84, '5_wm_pssm_Y': 85, '5_wm_pssm_V': 86, '7_wm_pssm_A': 87, '7_wm_pssm_R': 88, '7_wm_pssm_N': 89, 
    '7_wm_pssm_D': 90, '7_wm_pssm_C': 91, '7_wm_pssm_Q': 92, '7_wm_pssm_E': 93, '7_wm_pssm_G': 94, '7_wm_pssm_H': 95, 
    '7_wm_pssm_I': 96, '7_wm_pssm_L': 97, '7_wm_pssm_K': 98, '7_wm_pssm_M': 99, '7_wm_pssm_F': 100, '7_wm_pssm_P': 101, 
    '7_wm_pssm_S': 102, '7_wm_pssm_T': 103, '7_wm_pssm_W': 104, '7_wm_pssm_Y': 105, '7_wm_pssm_V': 106, '9_wm_pssm_A': 107, 
    '9_wm_pssm_R': 108, '9_wm_pssm_N': 109, '9_wm_pssm_D': 110, '9_wm_pssm_C': 111, '9_wm_pssm_Q': 112, '9_wm_pssm_E': 113, 
    '9_wm_pssm_G': 114, '9_wm_pssm_H': 115, '9_wm_pssm_I': 116, '9_wm_pssm_L': 117, '9_wm_pssm_K': 118, '9_wm_pssm_M': 119, 
    '9_wm_pssm_F': 120, '9_wm_pssm_P': 121, '9_wm_pssm_S': 122, '9_wm_pssm_T': 123, '9_wm_pssm_W': 124, '9_wm_pssm_Y': 125, 
    '9_wm_pssm_V': 126, 'AA': 127}

    """
    BIOLIP_FEATURE_NAMES = [
                'domain',
                'length',
                'ASA',
                'RSA',
                'PB',
                'PA',
                'PC',
                'pssm_A','pssm_R','pssm_N','pssm_D','pssm_C','pssm_Q','pssm_E','pssm_G','pssm_H','pssm_I',
                'pssm_L','pssm_K','pssm_M','pssm_F','pssm_P','pssm_S','pssm_T','pssm_W','pssm_Y','pssm_V',
                '3_wm_ASA','5_wm_ASA','7_wm_ASA','9_wm_ASA',
                '3_wm_RSA','5_wm_RSA','7_wm_RSA','9_wm_RSA',
                '3_wm_PB','5_wm_PB','7_wm_PB','9_wm_PB',
                '3_wm_PA','5_wm_PA','7_wm_PA','9_wm_PA',
                '3_wm_PC','5_wm_PC','7_wm_PC','9_wm_PC',
                '3_wm_pssm_A','3_wm_pssm_R','3_wm_pssm_N','3_wm_pssm_D','3_wm_pssm_C','3_wm_pssm_Q','3_wm_pssm_E',
                '3_wm_pssm_G','3_wm_pssm_H','3_wm_pssm_I','3_wm_pssm_L','3_wm_pssm_K','3_wm_pssm_M','3_wm_pssm_F',
                '3_wm_pssm_P','3_wm_pssm_S','3_wm_pssm_T','3_wm_pssm_W','3_wm_pssm_Y','3_wm_pssm_V',
                '5_wm_pssm_A','5_wm_pssm_R','5_wm_pssm_N','5_wm_pssm_D','5_wm_pssm_C','5_wm_pssm_Q','5_wm_pssm_E',
                '5_wm_pssm_G','5_wm_pssm_H','5_wm_pssm_I','5_wm_pssm_L','5_wm_pssm_K','5_wm_pssm_M','5_wm_pssm_F',
                '5_wm_pssm_P','5_wm_pssm_S','5_wm_pssm_T','5_wm_pssm_W','5_wm_pssm_Y','5_wm_pssm_V',
                '7_wm_pssm_A','7_wm_pssm_R','7_wm_pssm_N','7_wm_pssm_D','7_wm_pssm_C','7_wm_pssm_Q','7_wm_pssm_E',
                '7_wm_pssm_G','7_wm_pssm_H','7_wm_pssm_I','7_wm_pssm_L','7_wm_pssm_K','7_wm_pssm_M','7_wm_pssm_F',
                '7_wm_pssm_P','7_wm_pssm_S','7_wm_pssm_T','7_wm_pssm_W','7_wm_pssm_Y','7_wm_pssm_V',
                '9_wm_pssm_A','9_wm_pssm_R','9_wm_pssm_N','9_wm_pssm_D','9_wm_pssm_C','9_wm_pssm_Q','9_wm_pssm_E',
                '9_wm_pssm_G','9_wm_pssm_H','9_wm_pssm_I','9_wm_pssm_L','9_wm_pssm_K','9_wm_pssm_M','9_wm_pssm_F',
                '9_wm_pssm_P','9_wm_pssm_S','9_wm_pssm_T','9_wm_pssm_W','9_wm_pssm_Y','9_wm_pssm_V',
                'AA',
                ]
    
    GROUPS_BY_WM = {
                'domain': ['domain'],
                'length': ['length'],
                'ASA': ['ASA'],
                'RSA': ['RSA'],
                'PB': ['PB'],
                'PA': ['PA'],
                'PC': ['PC'],
                'pssm_A': ['pssm_A'],
                'pssm_R': ['pssm_R'],
                'pssm_N': ['pssm_N'],
                'pssm_D': ['pssm_D'],
                'pssm_C': ['pssm_C'],
                'pssm_Q': ['pssm_Q'],
                'pssm_E': ['pssm_E'],
                'pssm_G': ['pssm_G'],
                'pssm_H': ['pssm_H'],
                'pssm_I': ['pssm_I'],
                'pssm_L': ['pssm_L'],
                'pssm_K': ['pssm_K'],
                'pssm_M': ['pssm_M'],
                'pssm_F': ['pssm_F'],
                'pssm_P': ['pssm_P'],
                'pssm_S': ['pssm_S'],
                'pssm_T': ['pssm_T'],
                'pssm_W': ['pssm_W'],
                'pssm_Y': ['pssm_Y'],
                'pssm_V': ['pssm_V'],
                'WM_ASA': ['3_wm_ASA','5_wm_ASA','7_wm_ASA','9_wm_ASA'],
                'WM_RSA': ['3_wm_RSA','5_wm_RSA','7_wm_RSA','9_wm_RSA'],
                'WM_PB': ['3_wm_PB','5_wm_PB','7_wm_PB','9_wm_PB'],
                'WM_PA': ['3_wm_PA','5_wm_PA','7_wm_PA','9_wm_PA'],
                'WM_PC': ['3_wm_PC','5_wm_PC','7_wm_PC','9_wm_PC'],
                'WM_pssm_A': ['3_wm_pssm_A', '5_wm_pssm_A', '7_wm_pssm_A', '9_wm_pssm_A'], 
                'WM_pssm_R': ['3_wm_pssm_R', '5_wm_pssm_R', '7_wm_pssm_R', '9_wm_pssm_R'], 
                'WM_pssm_N': ['3_wm_pssm_N', '5_wm_pssm_N', '7_wm_pssm_N', '9_wm_pssm_N'], 
                'WM_pssm_D': ['3_wm_pssm_D', '5_wm_pssm_D', '7_wm_pssm_D', '9_wm_pssm_D'], 
                'WM_pssm_C': ['3_wm_pssm_C', '5_wm_pssm_C', '7_wm_pssm_C', '9_wm_pssm_C'], 
                'WM_pssm_Q': ['3_wm_pssm_Q', '5_wm_pssm_Q', '7_wm_pssm_Q', '9_wm_pssm_Q'], 
                'WM_pssm_E': ['3_wm_pssm_E', '5_wm_pssm_E', '7_wm_pssm_E', '9_wm_pssm_E'], 
                'WM_pssm_G': ['3_wm_pssm_G', '5_wm_pssm_G', '7_wm_pssm_G', '9_wm_pssm_G'], 
                'WM_pssm_H': ['3_wm_pssm_H', '5_wm_pssm_H', '7_wm_pssm_H', '9_wm_pssm_H'], 
                'WM_pssm_I': ['3_wm_pssm_I', '5_wm_pssm_I', '7_wm_pssm_I', '9_wm_pssm_I'], 
                'WM_pssm_L': ['3_wm_pssm_L', '5_wm_pssm_L', '7_wm_pssm_L', '9_wm_pssm_L'], 
                'WM_pssm_K': ['3_wm_pssm_K', '5_wm_pssm_K', '7_wm_pssm_K', '9_wm_pssm_K'], 
                'WM_pssm_M': ['3_wm_pssm_M', '5_wm_pssm_M', '7_wm_pssm_M', '9_wm_pssm_M'], 
                'WM_pssm_F': ['3_wm_pssm_F', '5_wm_pssm_F', '7_wm_pssm_F', '9_wm_pssm_F'], 
                'WM_pssm_P': ['3_wm_pssm_P', '5_wm_pssm_P', '7_wm_pssm_P', '9_wm_pssm_P'], 
                'WM_pssm_S': ['3_wm_pssm_S', '5_wm_pssm_S', '7_wm_pssm_S', '9_wm_pssm_S'], 
                'WM_pssm_T': ['3_wm_pssm_T', '5_wm_pssm_T', '7_wm_pssm_T', '9_wm_pssm_T'], 
                'WM_pssm_W': ['3_wm_pssm_W', '5_wm_pssm_W', '7_wm_pssm_W', '9_wm_pssm_W'], 
                'WM_pssm_Y': ['3_wm_pssm_Y', '5_wm_pssm_Y', '7_wm_pssm_Y', '9_wm_pssm_Y'], 
                'WM_pssm_V': ['3_wm_pssm_V', '5_wm_pssm_V', '7_wm_pssm_V', '9_wm_pssm_V'],
                'AA': ['AA'],
                }
    
    @classmethod
    def getFeatureNames(cls):
        if True:
            expFeatureNames = cls.BIOLIP_FEATURE_NAMES
        else:
            expFeatureNames = DatasetParams.FEATURE_COLUMNS
            
        return expFeatureNames
    
class Explanation(object):
    baseValues = None
    expValues = None
    
class PPIExplanationCls(object):
    @classmethod
    def setLogger(cls, loggerParam):
        global logger
        logger = loggerParam
    
    @classmethod
    def revertDict(cls, groups):
        revertedDict = dict(chain(*[zip(val, repeat(key)) for key, val in groups.items()]))
        return revertedDict
    
    @classmethod
    def groupFeatures(cls, expTestData, expValData, featureNames, groups):
        groupmap = cls.revertDict(groups)
        
        expTestTDF = pd.DataFrame(expTestData, columns=pd.Index(featureNames, name='findx')).T
        expTestTDF['group'] = expTestTDF.reset_index().findx.map(groupmap).values
        groupedExpTestDF = expTestTDF.groupby('group').mean().T
        
        expValTDF = pd.DataFrame(expValData, columns=pd.Index(featureNames, name='findx')).T
        expValTDF['group'] = expValTDF.reset_index().findx.map(groupmap).values
        groupedExpValDF = expValTDF.groupby('group').sum().T
        
        return groupedExpTestDF, groupedExpValDF
    
    @classmethod
    def getRandomSamples(cls, samples, randSize):
        numSampleInds = samples.shape[0]
        randSampleInds = np.random.choice(numSampleInds, size=randSize, replace=False)
        randSamples = samples[randSampleInds,:,:]
        return randSamples
    
    @classmethod
    def getAAIndsAndMask(cls):
        aaEndInd = DatasetParams.PROT_RESIDU_ENCODING_DIM
        aaInds = np.arange(ExplanationParams.AA_START_INDX, ExplanationParams.AA_START_INDX+aaEndInd)
        lenFeatures = DatasetParams.getFeaturesDim()
        mask = np.ones(lenFeatures, dtype=bool)
        mask[aaInds,] = False
        return aaInds, mask
    
    @classmethod    
    def sumupOnehotShapValues(cls, shapValues):
        aaInds, mask = cls.getAAIndsAndMask()
        aaShapValues = shapValues[:,aaInds]
        aaSummedShapValues = np.sum(aaShapValues, axis=1)
        #print('aaSummedShapValues.shape: ', aaSummedShapValues.shape)
        aaSummedShapValues = np.reshape(aaSummedShapValues, (aaSummedShapValues.shape[0],1))
        otherShapValues = shapValues[:,mask] 
        shapValues = np.insert(otherShapValues, [ExplanationParams.AA_START_INDX], aaSummedShapValues, axis=1)
        #print('explanation.shape-2: ', explanation.shape)
        return shapValues
    
    @classmethod    
    def replaceOnehotWithAA(cls, testX):
        aaInds, mask = cls.getAAIndsAndMask()
        aaOnehotValues = testX[:,aaInds]
        #np.where returns a tuple (arr-for-dim-1, arr-for-dim-2). We need the second array from the tuple.
        aaValues = (np.where(aaOnehotValues==1))[1]    #get the index of 1, which is the index of an AA in the AA-list.
        #aaValues = np.take(DatasetParams.AA_LIST, aaValues)    #could not convert string to float
        aaValues = np.reshape(aaValues, (aaValues.shape[0],1))
        otherValues = testX[:,mask]
        print('otherValues.shape: ', otherValues.shape, 'aaValues.shape: ', aaValues.shape)
        testX = np.insert(otherValues, [ExplanationParams.AA_START_INDX], aaValues, axis=1)
        return testX
    
    @classmethod
    def prepareBackgroundData(cls):
        #trainX, valX, trainY, valY = PPIDatasetCls.makeDataset(DatasetParams.EXPR_TRAINING_FILE, TrainingParams.INPUT_SHAPE, TrainingParams.LABEL_SHAPE, True)
        _, trainX, _, trainY = PPIDatasetCls.makeDataset(DatasetParams.EXPR_TESTING_FILE_SET[1], TrainingParams.INPUT_SHAPE, TrainingParams.LABEL_SHAPE, False)
        
        backgroundData = cls.getRandomSamples(trainX, ExplanationParams.NUM_BKS)
        return backgroundData
    
    @classmethod
    def prepareExplanation(cls, backgroundData):
        def evalModel(shapX):
            #print('@@@@@@@', shapX.shape)
            #shapX = np.reshape(shapX, modelInputShape)
            modelInputShape = (shapX.shape[0]//protLen, protLen, shapX.shape[1])
            #modelInputShape = shapX.shape
            #print('------',modelInputShape)
            shapX = np.reshape(shapX, modelInputShape)
            y_pred = model.predict(shapX, batch_size=None)
            #y_true, y_pred = PPILossCls.maskPadTargetTensor(backgroundY, y_pred)
            y_pred = y_pred.flatten()
            #y_pred = np.reshape(y_pred, (y_pred.shape[0], y_pred.shape[1]))
            #print('y_pred.shape: ', y_pred.shape)
            return y_pred
        
        def changeModel(model):
            #model.summary()
            firstLayer = model.get_layer(index=0)
            #print('firstLayer.input.shape: ', firstLayer.input.shape)
            protImg = firstLayer.input
            lastLayer = model.get_layer(index=-1)
            x = Flatten()(lastLayer.output)
            model = Model(inputs=protImg, outputs=x)
            model.summary()
            return model
        
        model = load_model(TrainingParams.MODEL_SAVE_FILE, compile=False, custom_objects=TrainingParams.CUSTOM_OBJECTS)
        
        """
        - shap creates all subsets of the feature-set. It can be thousands (2^M where M is the set of features).
        - shap  
        """
        
        if ExplanationParams.EXP_TYPE == ExpTypes.KERNEL:
            protLen = backgroundData.shape[1]
            shapInputShape = (ExplanationParams.NUM_BKS*protLen, backgroundData.shape[2])            #(5850, 28)
            #shapInputShape = backgroundData.shape
            backgroundData = np.reshape(backgroundData, shapInputShape)
            #print('=====',backgroundData.shape)
            explainer = shap.KernelExplainer(evalModel, backgroundData) #not yet supported (0.39) with new API.
        elif ExplanationParams.EXP_TYPE == ExpTypes.PERM:
            protLen = backgroundData.shape[1]
            shapInputShape = (ExplanationParams.NUM_BKS*protLen, backgroundData.shape[2])            #(5850, 28)
            #shapInputShape = backgroundData.shape
            backgroundData = np.reshape(backgroundData, shapInputShape)
            #print('=====',backgroundData.shape)
            explainer = shap.PermutationExplainer(evalModel, backgroundData) 
        elif ExplanationParams.EXP_TYPE == ExpTypes.DEEP:
            #put this statement otherwise we get: LookupError: gradient registry has no entry for: shap_AddV2
            #shap.explainers._deep.deep_tf.op_handlers["shap_AddV2"] = shap.explainers._deep.deep_tf.passthrough
            # problem: ||The model output must be a vector or a single value!|| but the shape our model's output is (#samples, 1024, 1).
            #print('=====',backgroundData.shape)
            #backgroundData = shap.maskers.Independent(backgroundData, max_samples=5) #error
            deepModel = changeModel(model)
            explainer = shap.DeepExplainer(deepModel, backgroundData)
            #explainer = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
        
        return explainer
    
    @classmethod 
    def prepareTestData(cls, testingFile):
        _, testX, _, testY = PPIDatasetCls.makeDataset(testingFile, TrainingParams.INPUT_SHAPE, TrainingParams.LABEL_SHAPE, False)
        
        expTestData = cls.getRandomSamples(testX, ExplanationParams.NUM_TSTS)
        if ExplanationParams.EXP_TYPE == ExpTypes.KERNEL:
            shapInputShape = (ExplanationParams.NUM_TSTS*expTestData.shape[1],expTestData.shape[2])                        #(1170, 28)
            expTestData = np.reshape(expTestData, shapInputShape)
        elif ExplanationParams.EXP_TYPE == ExpTypes.PERM:
            shapInputShape = (ExplanationParams.NUM_TSTS*expTestData.shape[1],expTestData.shape[2])                        #(1170, 28)
            expTestData = np.reshape(expTestData, shapInputShape) 
        elif ExplanationParams.EXP_TYPE == ExpTypes.DEEP:
            pass
        
        return expTestData
    
    @classmethod
    def getExpFileName(cls, testingLabel):
        expFileName = ExplanationParams.MODEL_EXP_FILE.format(TrainingParams.ALGRITHM_NAME, testingLabel)
        return expFileName
    
    @classmethod
    def loadExplanation(cls, testingLabel):
        expFileName = cls.getExpFileName(testingLabel)
        explanation = joblib.load(expFileName)
        return explanation
    
    @classmethod
    def saveExplanation(cls, explanation, testingLabel):
        expFileName = cls.getExpFileName(testingLabel)
        joblib.dump(explanation, expFileName, compress=0)
        return expFileName
    
    @classmethod    
    def doExplanation(cls, expTestData, explainer, testingFile):
        testingDateTime = datetime.datetime.now().strftime("%d-%m-%Y#%H:%M:%S")
        logger.info("## Explaining model " + TrainingParams.ALGRITHM_NAME + ' on ' + testingFile + " at: " + str(testingDateTime) + " ##")
        testingStartTime = time.time()
        
        if ExplanationParams.EXP_TYPE == ExpTypes.KERNEL:
            #old API, buggy
            explanation = Explanation()
            explanation.expValues = explainer.shap_values(expTestData,nsamples=500)  #only _kernel
            explanation.baseValues = explainer.expected_value
        elif ExplanationParams.EXP_TYPE == ExpTypes.PERM:
            #new API, but it's very buggy.
            #Note: explainer object changes heavily after applying it on a test dataset. So, we have to return it to the caller 
            explainer = explainer(expTestData)
            explanation = explainer
        elif ExplanationParams.EXP_TYPE == ExpTypes.DEEP:
            #tf.compat.v1.disable_eager_execution()  #'TFDeep' object has no attribute 'between_tensors'
            #explainer = explainer(expTestData)      #'Deep' object has no attribute 'masker'
            #explanation = explainer
            
            explanation = Explanation()
            explanation.expValues = explainer.shap_values(expTestData)  #infinite loop 
            explanation.baseValues = explainer.expected_value
            print('NNNNNNNNNNNN: not yet implemented')
        
        testingEndTime = time.time()
        logger.info("Explanation time: {}".format(datetime.timedelta(seconds=testingEndTime-testingStartTime)))
        
        return explanation
    
    @classmethod
    def preparePlotData(cls, expTestData, expValData, featureNames):
        ExplanationParams.INST_IND = random.randint(0, ExplanationParams.NUM_TSTS-1)
        ExplanationParams.AA_START_INDX = expTestData.shape[-1] - DatasetParams.PROT_RESIDU_ENCODING_DIM 
        newExpTestData = cls.replaceOnehotWithAA(expTestData)
        newExpValData = cls.sumupOnehotShapValues(expValData)
        if ExplanationParams.GROUP_BY_TYPE == GroupByTypes.BY_ALL:
            newFeatureNames = featureNames
        else:
            if ExplanationParams.GROUP_BY_TYPE == GroupByTypes.BY_WM:
                groupByType = ExplanationParams.GROUPS_BY_WM
            groupedExpTestDF, groupedExpValDF = cls.groupFeatures(newExpTestData, newExpValData, featureNames, groupByType)
            newExpTestData = groupedExpTestDF.values
            newExpValData = groupedExpValDF.values
            newFeatureNames = groupedExpValDF.columns
        newExpTestData = np.around(newExpTestData, decimals=3)
        newExpValData = np.around(newExpValData, decimals=3)
        
        return newExpTestData, newExpValData, newFeatureNames
    
    @classmethod
    def plotExplanation(cls, expTestData, explanation, expFileName):
        def saveExpFig(fig, figType):
            fig = plt
            figFile = expFileName.replace('EXP-', 'EXP-'+figType+'-')
            figFile = figFile.replace('.exp', '.png')
            fig.savefig(figFile, format="png", dpi=150, bbox_inches='tight')
            fig.clf()
            return
        
        def plotShapValues(expTestData, expValData, featureNames, baseValues):
            #testInst = expTestData[[ExplanationParams.INST_IND]]        #(128,)
            testInst = expTestData[ExplanationParams.INST_IND]           #(1,128)
            #print('testInst.shape: ', testInst.shape)
            
            fig = shap.summary_plot(shap_values=expValData, max_display=ExplanationParams.MAX_FEATURES, 
                                    features=expTestData, 
                                    feature_names=featureNames, show=False)
            saveExpFig(fig, 'SUMP')
            """
            fig = shap.plots.force(base_value=baseValues, shap_values=expValData[ExplanationParams.INST_IND], 
                                   features=testInst, feature_names=featureNames, figsize=(20,3), matplotlib=True, show=False)
            saveExpFig(fig, 'FORCE')
            fig = shap.summary_plot(shap_values=expValData, max_display=ExplanationParams.MAX_FEATURES, 
                                    features=expTestData, 
                                    feature_names=featureNames, show=False)
            saveExpFig(fig, 'SUMP')
            fig = shap.summary_plot(shap_values=expValData, max_display=ExplanationParams.MAX_FEATURES, 
                                    features=expTestData, 
                                    plot_type="bar", 
                                    feature_names=featureNames, show=False)
            saveExpFig(fig, 'BARP')
            fig = shap.plots._waterfall.waterfall_legacy(expected_value=baseValues, 
                                    shap_values=expValData[ExplanationParams.INST_IND], 
                                    features=testInst, max_display=ExplanationParams.MAX_FEATURES,
                                    feature_names=featureNames, show=False)
            saveExpFig(fig, 'WATER')
            fig = shap.plots._scatter.dependence_legacy(ind=ExplanationParams.SELECTED_FEATURE,
                                    shap_values=expValData, 
                                    features=expTestData, 
                                    #interaction_index="auto",
                                    interaction_index=ExplanationParams.DEPENDENT_FEATURE,
                                    feature_names=featureNames, show=False)
            saveExpFig(fig, 'DEPP')
            fig = shap.plots._bar.bar_legacy(shap_values=expValData[ExplanationParams.INST_IND], max_display=ExplanationParams.MAX_FEATURES, 
                                    features=testInst, 
                                    feature_names=featureNames, show=False)
            saveExpFig(fig, 'BAR')
            
            shapExp = shap.Explanation(values=expValData, base_values=baseValues, data=expTestData, feature_names=featureNames)
            fig = shap.plots.heatmap(shap_values=shapExp, max_display=ExplanationParams.MAX_FEATURES, show=False)
            saveExpFig(fig, 'HEATP')
            """
            return
        
        def plotShapExplainer(explainer, expTestData, expValData, featureNames):
            explainer.data = expTestData
            explainer.values = expValData
            explainer.feature_names = featureNames
            
            testInstance = explainer[ExplanationParams.INST_IND,:]
            testInstance.feature_names = featureNames
            
            fig = shap.plots.heatmap(shap_values=explainer, max_display=ExplanationParams.MAX_FEATURES, show=False)
            saveExpFig(fig, 'HEATP')
            #fig = shap.plots.scatter(shap_values=explainer[:,FOCUS_FEATURE_IND], color=explainer, show=False)
            #!fig = shap.plots.scatter(shap_values=explainer[:, explainer.abs.mean(0).argsort[-1]], show=False)
            #saveExpFig(fig, 'SCATP')
            fig = shap.plots.waterfall(shap_values=testInstance, max_display=ExplanationParams.MAX_FEATURES, show=False)
            saveExpFig(fig, 'WATER')
            #!fig = shap.plots.waterfall(shap_values=explainer, max_display=ExplanationParams.MAX_FEATURES, show=False)
            #saveExpFig(fig, 'WATERP')
            fig = shap.plots.force(base_value=explainer.base_values[ExplanationParams.INST_IND], 
                                   shap_values=explainer.values[ExplanationParams.INST_IND], 
                                   features=explainer.data[ExplanationParams.INST_IND],
                                   feature_names=explainer.feature_names, figsize=(20,3), matplotlib=True, show=False)
            saveExpFig(fig, 'FORCE')
            #!shap.plots.force(base_value=explainer.base_values, shap_values=explainer.values, 
            #               features=explainer.data,
            #               feature_names=DatasetParams.FEATURE_COLUMNS, figsize=(20,3), matplotlib=True, show=False)
            #saveExpFig(fig, 'FORCEP')
            fig = shap.plots.beeswarm(shap_values=explainer, max_display=ExplanationParams.MAX_FEATURES, show=False)
            saveExpFig(fig, 'BEESP')
            fig = shap.plots.bar(shap_values=explainer, max_display=ExplanationParams.MAX_FEATURES, show=False)
            saveExpFig(fig, 'BARP')
            fig = shap.plots.bar(shap_values=testInstance, max_display=ExplanationParams.MAX_FEATURES, show=False)
            saveExpFig(fig, 'BAR')
            
            return
        
        testingDateTime = datetime.datetime.now().strftime("%d-%m-%Y#%H:%M:%S")
        logger.info("## Plotting explanation of model " + TrainingParams.ALGRITHM_NAME + ' on ' + expFileName + " at: " + str(testingDateTime) + " ##")
        logger.info("Chosen example index: " + str(ExplanationParams.INST_IND) + ' and selected feature: ' + ExplanationParams.SELECTED_FEATURE)
        
        if True:    #here we can choose for testing dataset
            featureNames = ExplanationParams.getFeatureNames()
        
        if ExplanationParams.EXP_TYPE == ExpTypes.KERNEL:
            expTestData, expValData, featureNames = cls.preparePlotData(expTestData, explanation.expValues, featureNames)
            plotShapValues(expTestData, expValData, featureNames, explanation.baseValues)
        elif ExplanationParams.EXP_TYPE == ExpTypes.PERM:
            explainer = explanation
            expTestData, expValData, featureNames = cls.preparePlotData(expTestData, explainer.values, featureNames)
            plotShapExplainer(explainer, expTestData, expValData, featureNames)
        elif ExplanationParams.EXP_TYPE == ExpTypes.DEEP:
            print('NNNNNNNNNNNN: not yet implemented')
        
        return
    
    @classmethod
    def explainModel(cls):
        #tf.compat.v1.disable_v2_behavior()    #no effect
        
        if not ExplanationParams.USE_SAVED_EXPS:
            backgroundData = cls.prepareBackgroundData()
            explainer = cls.prepareExplanation(backgroundData)
            #for i in range(len(DatasetParams.EXPR_TESTING_FILE_SET)):
            for i in range(0,1):
                testingFile = DatasetParams.EXPR_TESTING_FILE_SET[i]
                testingLabel = DatasetParams.EXPR_TESTING_LABELS[i]
                expTestData = cls.prepareTestData(testingFile)
                explanation = cls.doExplanation(expTestData, explainer, testingFile)
                expFileName = cls.saveExplanation(explanation, testingLabel)
                cls.plotExplanation(expTestData, explanation, expFileName)
        else:
            #for i in range(len(DatasetParams.EXPR_TESTING_FILE_SET)):
            for i in range(0,1):
                testingFile = DatasetParams.EXPR_TESTING_FILE_SET[i]
                testingLabel = DatasetParams.EXPR_TESTING_LABELS[i]
                expFileName = cls.getExpFileName(testingLabel)
                explanation = cls.loadExplanation(testingLabel)
                expTestData = cls.prepareTestData(testingFile)
                cls.plotExplanation(expTestData, explanation, expFileName)
        return    
