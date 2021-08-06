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
                                           ReLU, LeakyReLU, PReLU, ELU

from importlib import import_module

from PPILogger import PPILoggerCls
from PPIDataset import PPIDatasetCls, DatasetParams
from PPILoss import PPILossCls, LossParams
from PPITrainTest import PPITrainTestCls, TrainingParams
from PPIParams import PPIParamsCls
from PPIPredPlot import PPIPredPlotCls

rnn = import_module('rnn-ppi-keras', package='../rnn-ppi')
ann = import_module('ann-ppi-keras', package='../ann-ppi')
dnet = import_module('dnet-XD-ppi-keras', package='../dnet-ppi')
rnet = import_module('rnet-XD-ppi-keras', package='../rnet-ppi')
cnn_rnn = import_module('cnn-rnn-XD-ppi-keras', package='../cnn-rnn-ppi')
unet = import_module('unet-XD-ppi-keras', package='../unet-ppi')
algs = [
        rnn, 
        ann, 
        dnet, 
        rnet, 
        cnn_rnn, 
        unet
        ]
# This must be dynamic but for the time being we do it static.
modelsNames = ['rnn_p' , 
               'ann_p', 
               'dnet_p', 
               'rnet_p', 
               'cnet_p', 
               'unet_p', 
               'ensnet_p']

#algs = [ann,rnet,cnn_rnn,unet]       #best-1 only for Epitope dataset

class AlgParams:
    ALGRITHM_NAME = "ensnet-ppi"
    logger = None
    DatasetParams.USE_COMET = False #True
    
    datasetLabel = 'Biolip_N'
    dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    
    PLOT_METRICS_PER_PROT = False #True     #put it on True if you don't want to do anything except for plotting metrics per protein 
    #PLOT_TYPE_METRICS_PER_PROT = PPIPredPlotCls.VIOLIN_MCC
    PLOT_TYPE_METRICS_PER_PROT = PPIPredPlotCls.SCATTER_AVG_MCC
    
    #Keras-ANN: ZK448_P=[MCC: 25.93% AUC: 71.89% AP: 38.42%] and BioDL_P=[MCC: 24.90% AUC: 75.56% AP: 30.22%]
    #DIRICHLET: ZK448_P=[MCC: 25.84% AUC: 71.95% AP: 38.32%] and BioDL_P=[MCC: 24.92% AUC: 75.58% AP: 30.14%]
    #RF and XGBOOST score lower.
    
    USE_SAVED_ENS_DATA = False #True        #use the already save predictions, which are train-set and test-set of ensnet
    GEN_COMB_ROC = False                    #set it on True to generate combined ROC plots
    
    ENS_DIRICHLET_MODEL = False #True
    ENS_RF_MODEL = False #True
    ENS_XGBOOST_MODEL = False #True
    
    ONLY_TEST = True #False
    RANDOM_SAMPLE_TEST = False #True     #set it on True if you want to take just a random sample of your test set and not the whole data set
    
    ANN_MODEL = True #False
    USE_AVERAGE = False #True
    
    if USE_AVERAGE:
        width = 1
    else:
        width = len(algs)
    
    TrainingParams.ENSEMBLE_ARCHS = [alg.AlgParams.ALGRITHM_NAME for alg in algs]
    
    LABEL_DIM = 1
    if ANN_MODEL:
        PROT_IMAGE_H, PROT_IMAGE_W = 1,width
    else:
        PROT_IMAGE_H, PROT_IMAGE_W = 1024,width
    INPUT_SHAPE = None  #must be determined in init.
    LABEL_SHAPE = None
    
    CNN_KERNEL_SIZE = 7 
    INIT_CNN_DILATION_SIZE = 1
    CNN_CHANNEL_SIZE = 128 
    NUM_BLOCK_REPEATS = 8
    
    @classmethod
    def setShapes(cls):
        cls.INPUT_SHAPE = (cls.PROT_IMAGE_H, cls.PROT_IMAGE_W)   
        cls.LABEL_SHAPE = (cls.PROT_IMAGE_H, cls.LABEL_DIM)
        PPIParamsCls.setShapeParams(cls.INPUT_SHAPE, cls.LABEL_SHAPE)   
        return
    
    @classmethod
    def initAlgParams(cls, dsParam=dataset, dsLabelParam=datasetLabel):
        PPIParamsCls.setInitParams(cls.ALGRITHM_NAME,dsParam=dsParam, dsLabelParam=dsLabelParam)
        cls.setShapes()
        return

def makeDenseBlock(x, denseSize):
    x = Dense(denseSize)(x)
    x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    x = BatchNormalization()(x)    
    x = TrainingParams.ACTIVATION_FUN()(x)
    return x

def makeAnnModel():
    LossParams.USE_WEIGHTED_LOSS = False
    DatasetParams.USE_DOWN_SAMPLING = True
    DatasetParams.DOWN_SAMPLING_THRESHOLD = 0.3
    
    HIDDEN_LAYER_1_SIZE = 64 #16 #512 #16 #512
    HIDDEN_LAYER_2_SIZE = 16 #128
    HIDDEN_LAYER_3_SIZE = 32
    HIDDEN_LAYER_4_SIZE = 8

    LAYERS_SIZES = [
                    #HIDDEN_LAYER_1_SIZE, 
                    #HIDDEN_LAYER_2_SIZE, 
                    #HIDDEN_LAYER_3_SIZE, 
                    #HIDDEN_LAYER_4_SIZE,
                   ] 
    
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    for layer in range(len(LAYERS_SIZES)):
        x = makeDenseBlock(x, LAYERS_SIZES[layer])
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=protImg, outputs=x)
    return model

def makeDnetConv(x, channelSize,  dilationRate):
    ki = TrainingParams.KERNAL_INITIALIZER() 
    ub = TrainingParams.USE_BIAS 
    #kr = l1_l2(TrainingParams.REG_LAMDA, TrainingParams.REG_LAMDA)
    #br = l1_l2(TrainingParams.REG_LAMDA, TrainingParams.REG_LAMDA)
    x = Conv1D(channelSize, AlgParams.CNN_KERNEL_SIZE, dilation_rate=dilationRate, 
               #kernel_regularizer=kr, bias_regularizer=br,
               padding='same', use_bias=ub, kernel_initializer=ki)(x)
    x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    x = BatchNormalization()(x)    
    x = TrainingParams.ACTIVATION_FUN()(x)
    return x 

def makeDnetModel():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg

    channelSize = 64 
    dilationRate = AlgParams.INIT_CNN_DILATION_SIZE
    x = makeDnetConv(x, channelSize, dilationRate) 
    channelSize = 128
    dilationRate = dilationRate * 2
    x = makeDnetConv(x, channelSize, dilationRate) 
    channelSize = 128
    dilationRate = dilationRate * 2
    x = makeDnetConv(x, channelSize, dilationRate) 
    channelSize = 64
    dilationRate = dilationRate * 2
    x = makeDnetConv(x, channelSize, dilationRate) 
    channelSize = 32
    dilationRate = dilationRate * 2
    x = makeDnetConv(x, channelSize, dilationRate) 
    
    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
    model = Model(inputs=protImg, outputs=x)
    return model 

def makeActNormBlock(x):
    x = Dropout(TrainingParams.DROPOUT_RATE)(x)
    x = BatchNormalization()(x)    
    x = TrainingParams.ACTIVATION_FUN()(x)
    return x

def makeRnetConv(x, ks):
    ki = TrainingParams.KERNAL_INITIALIZER() 
    ub = TrainingParams.USE_BIAS 
    cs = AlgParams.CNN_CHANNEL_SIZE
    
    x = Conv1D(filters=cs, kernel_size=ks, strides=1, padding='same', use_bias=ub, kernel_initializer=ki)(x)
    return x

def makeRnetResBlock(x):
    res = x
    x = makeActNormBlock(x)
    x = makeRnetConv(x, 5)
    x = makeActNormBlock(x)
    x = makeRnetConv(x, 3)
    x = Add()([res,x])
    return x  

def makeRnetModel():
    protImg = Input(shape=AlgParams.INPUT_SHAPE)
    x = protImg
    
    x = makeRnetConv(x, 3)
    for i in range(AlgParams.NUM_BLOCK_REPEATS):
        x = makeRnetResBlock(x)
    
    x = makeActNormBlock(x)    
    x = TimeDistributed(Dense(1, activation='sigmoid'))(x)
        
    model = Model(inputs=protImg, outputs=x)
    return model 

def makeEnsModel():
    if AlgParams.ANN_MODEL:
        model = makeAnnModel()
    else:
        #model = makeDnetModel()
        model = makeRnetModel()
    return model

def performTraining(ensTrainDataset):
    #TrainingParams.NUM_EPOCS = 120
    #LossParams.setLossFun(LossParams.MEAN_SQUARED)
    if TrainingParams.ENS_DIRICHLET_MODEL:
        model = None
    else:
        model = makeEnsModel()
    TrainingParams.persistParams(sys.modules[__name__])
    PPITrainTestCls().trainModel(model, ensTrainDataset)
    return

def performTesting(modelsResults):
    DatasetParams.USE_DOWN_SAMPLING = False
    TrainingParams.GEN_METRICS_PER_PROT = False #True
    ensResults = PPITrainTestCls().testModel(modelsResults)
    return ensResults

def prepareEnsembleData(datasetIndex):
    # y_trues = [test-set-1=(64,1024,1), test-set-2=(44,1024,1)] of one algorithm (e.g., rnn-ppi)
    stacked_y_true,stacked_y_pred = None, None
    for alg in algs:
        TrainingParams.resetOutputFileNames()   #we set output-file-names again via initAlgParams
        arch = alg.AlgParams.ALGRITHM_NAME
        AlgParams.logger.info("======= ARCHITECTURE_NAME: " + arch)
        alg.AlgParams.initAlgParams(dsParam=AlgParams.dataset, dsLabelParam=AlgParams.datasetLabel, ensembleTesting=True)
        #cur_y_true,cur_y_pred = alg.performTesting()
        cur_y_true,cur_y_pred = PPITrainTestCls.testModelForEnsembl(datasetIndex)
        stacked_y_true,stacked_y_pred = PPITrainTestCls.prepareEnsembledPreds(stacked_y_true, stacked_y_pred, cur_y_true, cur_y_pred)
    TrainingParams.resetOutputFileNames()   #leave clean the output-file-names  
    
    inputData = stacked_y_pred
    labelData = stacked_y_true
    AlgParams.logger.info('=======inputData (preds): ' + str(inputData.shape))      #(84941,1,6)
    AlgParams.logger.info('=======labelData(trues): ' + str(labelData.shape))       #(84941,1,1)
    if AlgParams.USE_AVERAGE:
        inputData = PPIDatasetCls.averageData(inputData)
    if not AlgParams.ANN_MODEL: #change shape
        inputData = PPIDatasetCls.sliceData(inputData, AlgParams.INPUT_SHAPE, isLabel=False)
        labelData = PPIDatasetCls.sliceData(labelData, AlgParams.LABEL_SHAPE, isLabel=True)
    ensDataset = [inputData, labelData]
    return ensDataset

def setEnsembleModelType(bool):
    if AlgParams.ENS_DIRICHLET_MODEL:
        TrainingParams.ENS_DIRICHLET_MODEL = bool
    elif AlgParams.ENS_RF_MODEL:
        TrainingParams.ENS_RF_MODEL = bool
    elif AlgParams.ENS_XGBOOST_MODEL:
        TrainingParams.ENS_XGBOOST_MODEL = bool
    return    

def doEnsembling():
    def doEnsembleTesting(modelsResults):
        TrainingParams.USE_ENSEMBLE_TESTING = False
        AlgParams.initAlgParams()   #set ensemble dataset and output-file-names (including model-name)
        setEnsembleModelType(True)
        ensResults = performTesting(modelsResults)
        setEnsembleModelType(False)
        return ensResults
    
    def applyModelsOnFixedTestsets():
        TrainingParams.USE_ENSEMBLE_TESTING = True
        modelsResults = []
        for i in range(len(DatasetParams.EXPR_TESTING_FILE_SET)):
            if not AlgParams.USE_SAVED_ENS_DATA:
                modelsResult = prepareEnsembleData(i)    #these are predictions of the testing set
                PPITrainTestCls.saveEnsDataset(modelsResult, False, i)
            else:
                modelsResult = PPITrainTestCls.getEnsDataset(False, i)
            modelsResults.append(modelsResult)
        ensResults = doEnsembleTesting(modelsResults)
        return modelsResults, ensResults
    
    def applyModelsOnRandomTestsets():
        DatasetParams.VALIDATION_RATE = 0.5     #we misuse this to pass the fraction number for random sampling
        testIter = 10
        for i in range(testIter):
            TrainingParams.USE_ENSEMBLE_TESTING = True
            TrainingParams.USE_DATASET_PARTITIONING = True 
            modelsResult = prepareEnsembleData(0)    #these are predictions of the testing set
            DatasetParams.RANDOM_STATE = DatasetParams.RANDOM_STATE + 10    #all samples can be the same if random-state is the same
            TrainingParams.USE_DATASET_PARTITIONING = False
            modelsResults = [modelsResult]
            ensResults = doEnsembleTesting(modelsResults)
        return modelsResults, ensResults
    
    AlgParams.logger = PPIParamsCls.setLoggers(AlgParams.ALGRITHM_NAME, AlgParams.datasetLabel)
    testingDateTime = datetime.datetime.now().strftime("%d-%m-%Y#%H:%M:%S")
    AlgParams.logger.info("\n#### Ensemble-Testing " + AlgParams.datasetLabel + " at: " + str(testingDateTime) + " ####")
    testingStartTime = time.time()
    
    AlgParams.initAlgParams()   #set ensemble dataset 
    if not AlgParams.ONLY_TEST:
        TrainingParams.USE_ENSEMBLE_TRAINING = True
        TrainingParams.USE_DATASET_PARTITIONING = True    #must be dataset partitioned? So, you can pick up only the validation part.
        ensTrainDatasetLabel = AlgParams.datasetLabel
        #AlgParams.datasetLabel = 'Homo_Hetro'            #set here another dataset than defined in AlgParams
        #AlgParams.datasetLabel = 'Homo'
        
        if not AlgParams.USE_SAVED_ENS_DATA:
            ensTrainDataset = prepareEnsembleData(0)           #these are predictions of the training set
            PPITrainTestCls.saveEnsDataset(ensTrainDataset, True)
        else:
            ensTrainDataset = PPITrainTestCls.getEnsDataset(True)
        TrainingParams.USE_ENSEMBLE_TRAINING = False
        
        TrainingParams.USE_DATASET_PARTITIONING = False
        AlgParams.datasetLabel = ensTrainDatasetLabel
        
        AlgParams.initAlgParams()   #set ensemble dataset and output-file-names (including model-name) 
        setEnsembleModelType(True)
        performTraining(ensTrainDataset)
        setEnsembleModelType(False)
        TrainingParams.resetOutputFileNames()  #leave clean the output-file-names 
    
    if AlgParams.RANDOM_SAMPLE_TEST:
        applyModelsOnRandomTestsets()
    else:
        applyModelsOnFixedTestsets()
      
    if AlgParams.GEN_COMB_ROC:  #generate combined AUC-ROC plot from the predictions of all models, including ensnet.
        #modelsResults=[[(99318,1,6), (99318,1,1)],    [(64609,1,6), (64609,1,1)]]   #list of list of arrays; one list per test data set
        #ensResults=     [[(99318,),    (99318,)   ],    [(64609,),    (64609,)]]    #list of list of lists = [preds,trues]
        PPITrainTestCls().plotStackedMetrics(modelsNames, modelsResults, ensResults)
    
    testingEndTime = time.time()
    AlgParams.logger.info("Testing time: {}".format(datetime.timedelta(seconds=testingEndTime-testingStartTime)))
    
    return
    
def plotMetricsPerProt():
    AlgParams.logger = PPIParamsCls.setLoggers(AlgParams.ALGRITHM_NAME, AlgParams.datasetLabel)
    AlgParams.initAlgParams()
    modelsAlgs = TrainingParams.ENSEMBLE_ARCHS
    modelsAlgs.append(AlgParams.ALGRITHM_NAME)
    PPITrainTestCls().plotMetricsPerProt(modelsAlgs, modelsNames, AlgParams.PLOT_TYPE_METRICS_PER_PROT)
    return
      
if __name__ == "__main__":
    if AlgParams.PLOT_METRICS_PER_PROT:
        plotMetricsPerProt()
    else:
        doEnsembling() 
        