# make sure comet_ml is the first import (before all other Machine learning lib)
from comet_ml import Experiment

import random
import datetime
import time
import math
import inspect, re
import json
import joblib
from pprint import pprint
import pandas as pd
import numpy as np
import heapq as hq

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb

import tensorflow as tf
#import tensorflow.compat.v1 as tf

from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.layers import ReLU, PReLU, Flatten
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import plot_model, Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler

from treelib import Tree, Node

from PPIDataset import PPIDatasetCls, DatasetParams
from PPILoss import PPILossCls, LossParams
from PPIPredPlot import PPIPredPlotCls, PPIPredMericsCls

# you can't put these statements in a function. Setting seed doesn't have any effect any more.
SEED_VAL = 50 #7
#tf.enable_eager_execution()
#tf.set_random_seed(SEED_VAL)
tf.random.set_seed(SEED_VAL)
np.random.seed(SEED_VAL)
# we need to include these statements because of a bug in cuda-module for RTX gpus. There is no issue with GTX gpus.
"""
gpu_devices = tf.config.experimental.list_physical_devices('GPU') 
for device in gpu_devices: 
    tf.config.experimental.set_memory_growth(device, True)
"""
def doesTFUseGPU():
    if tf.test.gpu_device_name():
        print('###### Default GPU Device:{}'.format(tf.test.gpu_device_name()))
        print("###### Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
        return True
    else:
        print("$$$$$$ Info: No GPU version of TF is used.")
    return False 

def limitgpu(maxmem):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate a fraction of GPU memory
        try:
            for gpu in gpus:
                tf.config.experimental.set_virtual_device_configuration(gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=maxmem)])
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    return 

if doesTFUseGPU():
    memLimit = 6000 #6G
    #print('Limiting GPU memory use to: ', memLimit)
    #limitgpu(memLimit)  

logger = None

class MyPReLU(PReLU):
    def build(self, input_shape):
        #param_shape = tuple(1 for _ in range(len(input_shape)-1)) + input_shape[-1:]
        param_shape = None
        #print('======== param_shape: ', param_shape)
        self.alpha = self.add_weight(shape=param_shape,
                                     name='alpha',
                                     initializer=self.alpha_initializer,
                                     regularizer=self.alpha_regularizer,
                                     constraint=self.alpha_constraint)
        self.built = True

class TrainingParams(object):
    ENS_XGBOOST_MODEL = False
    ENS_DIRICHLET_MODEL = False
    ENS_RF_MODEL = False
    GEN_METRICS_PER_PROT = False
    USE_ENSEMBLE_TRAINING = False
    USE_ENSEMBLE_TESTING = False
    USE_DATASET_PARTITIONING = False
    ENSEMBLE_ARCHS = []
    ALGRITHM_NAME = None
    CUSTOM_OBJECTS = {}
    
    MODEL_DIR = "../models/{}/"
    MODEL_OUTPUT_DIR = "../models/{}/"
    METRICS_PER_PROT_FILE = "metrics-per-prot-{}.csv"
    MODEL_SAVE_FILE = "{}-model.hdf5"
    TB_STAT_DIR = "{}-TB-Stats"
    MODEL_PLOT_FILE = "{}-model-plot.png"
    MODEL_PRED_FILE = "preds-{}.csv"
    ENS_TRAIN_FILE = "ens_training_{}.csv"
    ENS_TEST_FILE = "ens_testing_{}.csv"
    MODEL_PARAMS_FILE = "{}-model-params.txt"
    
    SAVE_PRED_FILE = False
    INPUT_SHAPE = None      #train inputShape without sample-size
    LABEL_SHAPE = None      #train labelShape without sample-size   
    TRAIN_SHAPE = []        #[inputShape, labelShape] for train; each including sample-size
    VAL_SHAPE = []          #[inputShape, labelShape] for validation; each including sample-size
    TEST_SHAPE = []         #[testingLabel,[inputShape, labelShape]]; each including sample-size
    NUM_ITERATIONS = 0
    TRAIN_DURATION = None
    ENS_DATA_COLUMNS = []    #This is a dynamic list, depending on the number of algorithms used for ensembling
    
    ADAM_OPT = 0
    SGD_OPT = 1
    RMS_OPT = 2
    OPT_NAMES = ['ADAM', 'SGD', 'RMS']
    OPT_FUN = ADAM_OPT
    
    #GlorotUniform, GlorotNormal, he_uniform, he_normal, lecun_uniform, lecun_normal, Orthogonal, TruncatedNormal, VarianceScalings
    KERNAL_INITIALIZER = GlorotUniform
    #ReLU, LeakyReLU, PReLU, ELU
    ACTIVATION_FUN = ReLU
    USE_BIAS = True
    MODEL_CHECK_POINT_MODE = -1   # 1 for max-mode (val_auc) and -1 for min-mode (val_loss)
    EARLY_STOP_PATIENCE = 80
    USE_CLASS_WEIGHT = False
    USE_EARLY_STOPPING = False
    USE_REDUCE_LR = False
    USE_LR_SCHEDULER = False
    USE_TENSOR_BOARD = False
    VERBOSE_INDICATOR = 0 #2
    BATCH_SIZE = 8 #16 #32 #64 #4 #2 
    DROPOUT_RATE = 0.2
    LEARNING_RATE = 1e-4 #1e-3 #1e-5
    REG_LAMDA = 0.001
    DECAY_RATE1 = 0.9
    DECAY_RATE2 = 0.999
    MOMENTUM_RATE = 0.7 #0.2 #0.9 #0.5 #0.4 #1.5
    EPSILON_RATE = 1e-8 #1e-2 #1e-8
    NUM_EPOCS = 800 #400 #200
    
    ATT_ENCODER_MODEL = None
    ATT_DECODEL_MODEL = None
    USE_BIDIRECTIONAL_ATT = True
    ATT_BEAM_WIDTH = 1
    
    @classmethod
    def initKeras(cls, floatType):
        DatasetParams.FLOAT_TYPE = floatType
        K.set_floatx(floatType)
        return
    
    @classmethod
    def initExperiment(cls, dsLabelParam):
        DatasetParams.COMET_EXPERIMENT = Experiment(project_name=cls.ALGRITHM_NAME)
        DatasetParams.COMET_EXPERIMENT.disable_mp()
        exprName = cls.ALGRITHM_NAME.strip('-ppi') + '_' + dsLabelParam
        DatasetParams.COMET_EXPERIMENT.set_name(exprName)
        #DatasetParams.COMET_EXPERIMENT.log_asset_folder('../' + cls.ALGRITHM_NAME)
        #DatasetParams.COMET_EXPERIMENT.log_asset_folder('../utils')
        return
    
    @classmethod
    def resetOutputFileNames(cls):
        if DatasetParams.USE_USERDS_EVAL:
            modelDir = DatasetParams.PIPENN_HOME + 'models/all-models/'
        elif DatasetParams.USE_PIPENN_TEST:
            modelDir = "./models/"
        else:
            modelDir = "../models/"
            
        cls.MODEL_DIR = modelDir + "{}/"
        cls.MODEL_OUTPUT_DIR = modelDir + "{}/"
        cls.MODEL_SAVE_FILE = "{}-model.hdf5"
        cls.METRICS_PER_PROT_FILE = "metrics-per-prot-{}.csv"
        cls.TB_STAT_DIR = "{}-TB-Stats"
        cls.MODEL_PLOT_FILE = "{}-model-plot.png"
        cls.MODEL_PRED_FILE = "preds-{}.csv"
        cls.MODEL_PARAMS_FILE = "{}-model-params.txt"
        #cls.ENS_TRAIN_FILE = "ens_training_{}.csv"
        #cls.ENS_TEST_FILE = "ens_testing_{}.csv"
        return
    
    @classmethod
    def getUserDSModelFile(cls):
        modelDir = DatasetParams.PIPENN_HOME + 'models/all-models/'
        userDSModelDirDict = {
            'UserDS_P': modelDir + 'biolip-p/',
            'UserDS_S': modelDir + 'biolip-s/',
            'UserDS_N': modelDir + 'biolip-n/',
            'UserDS_A': modelDir + 'biolip-a/',
        }
        return userDSModelDirDict.get(DatasetParams.EXPR_DATASET_LABEL)
    
    @classmethod
    def setEnsOutputFileNames(cls, algorithmName):
        cls.resetOutputFileNames()
        if DatasetParams.USE_USERDS_EVAL:
            outputDir = DatasetParams.USERDS_OUTPUT_DIR.format(algorithmName)        
        else:
            outputDir = cls.MODEL_OUTPUT_DIR.format(algorithmName)
        
        cls.ENS_TRAIN_FILE = outputDir + cls.ENS_TRAIN_FILE
        cls.ENS_TEST_FILE = outputDir + cls.ENS_TEST_FILE
        return 
    
    @classmethod
    def setOutputFileNames(cls, algorithmName):
        print("## USE_USERDS_EVAL: ", DatasetParams.USE_USERDS_EVAL)
        cls.resetOutputFileNames()
        if DatasetParams.USE_USERDS_EVAL:
            modelDir = cls.getUserDSModelFile()
            outputDir = DatasetParams.USERDS_OUTPUT_DIR.format(algorithmName)        
        else:
            modelDir = cls.MODEL_DIR.format(algorithmName)
            outputDir = cls.MODEL_OUTPUT_DIR.format(algorithmName)
        
        print("## PIPENN model-dir: ", modelDir, " and output-dir: ", outputDir)
        # used for both training and testing
        cls.MODEL_SAVE_FILE = modelDir + cls.MODEL_SAVE_FILE.format(algorithmName)
        
        # used only for training
        cls.TB_STAT_DIR = outputDir + cls.TB_STAT_DIR.format(algorithmName)
        cls.MODEL_PARAMS_FILE = outputDir + cls.MODEL_PARAMS_FILE.format(algorithmName)
        cls.MODEL_PLOT_FILE = outputDir + cls.MODEL_PLOT_FILE.format(algorithmName)
        
        # the name of these output files depend on testing label which are set during eval of testing files. 
        # So, here we set only the correct directory.
        cls.METRICS_PER_PROT_FILE = outputDir + cls.METRICS_PER_PROT_FILE
        cls.MODEL_PRED_FILE = outputDir + cls.MODEL_PRED_FILE
        return
    
    @classmethod
    def persistParams(cls, algModule):
        algParams = inspect.getmembers(algModule, lambda a:not(inspect.isroutine(a)))
        trainParams = inspect.getmembers(cls, lambda a:not(inspect.isroutine(a)))
        datasetParams = inspect.getmembers(DatasetParams, lambda a:not(inspect.isroutine(a)))
        lossParams = inspect.getmembers(LossParams, lambda a:not(inspect.isroutine(a)))
        allParams = algParams + trainParams + datasetParams + lossParams
        #pattern = re.compile("^[A-Z]([A-Z]|[_]|[0-9])*[A-Z]$")
        #modelParams = [a for a in allParams if pattern.match(a[0])]
        modelParams = [a for a in allParams if not(a[0].startswith('__') and a[0].endswith('__'))]
        
        with open(cls.MODEL_PARAMS_FILE, mode='w', encoding='utf-8') as f:
            pprint(dict(modelParams), f)
            #json.dump(dict(allParams), f, sort_keys=True, indent=4)
        
        return
    
class Histories(Callback):
    valX, valY = None, None
    
    def setValset(self, valX, valY):
        self.valX, self.valY = valX, valY    
        
    def printYoloData(self):
        # record 205 from hetro_testing with features ('mean_H','ASA_H', 'ASA_q'): [0.77123529 0.47467237 0.35551524] 
        print(self.validation_data[0].shape)
        print(self.validation_data[1].shape)
        print(self.validation_data[0][0,1,0])
        print(self.validation_data[1][0,1,2,0,5:])  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
        print(self.validation_data[1][0,1,1,0,5:])  # [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
        
    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return
    
    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        valLoss = logs.get('val_loss')
        #self.printYoloData()
        
        y_true, y_pred = PPITrainTestCls.doEvaluation(self.model, self.valX, self.valY)
        PPILossCls.logTrainingMetrics(epoch, loss, valLoss, y_true, y_pred)
        #PPILossCls.logTestingMetrics(y_true, y_pred)
        
        #print("################ val_auc: ", logs.get('val_auc'))
        
        return

    def on_batch_begin(self, batch, logs={}):
        #print("################ batch-begin")
        return

    def on_batch_end(self, batch, logs={}):
        #print("=============== batch-end")
        return

class ModelCheckPointAtMaxAuc(Callback):
    def __init__(self):
        super(ModelCheckPointAtMaxAuc, self).__init__()
        
        self.valX, self.valY = None, None
        if TrainingParams.MODEL_CHECK_POINT_MODE == -1:
            self.minDelta = 0.005
        else:
            self.minDelta = 0.001
        self.patience = TrainingParams.EARLY_STOP_PATIENCE   
        self.bestWeights = None    # bestWeights to store the weights at which the max auc occurs.
        self.bestEpoch = None
        return
    
    def setValset(self, valX, valY):
        self.valX, self.valY = valX, valY   
        return
        
    def on_train_begin(self, logs=None):
        self.wait = 0               # The number of epoch it has waited when auc is no longer max.
        self.stoppedEpoch = 0       # The epoch the training stops at.
        
        if TrainingParams.MODEL_CHECK_POINT_MODE == -1:
            self.bestAuc = np.inf
        else:
            self.bestAuc = np.NINF
        return
    
    #def on_batch_begin(self, batch, logs={}):
    #    print('%%%%%%%%%% batch begin %%%%%%%%%')
    #    return
    
    def on_batch_end(self, batch, logs={}):
        #print('======== batch end ==========')
        TrainingParams.NUM_ITERATIONS = TrainingParams.NUM_ITERATIONS + 1 
        if DatasetParams.COMET_EXPERIMENT is not None:
            metrics = {
                    "num_iterations": TrainingParams.NUM_ITERATIONS,
                }
            DatasetParams.COMET_EXPERIMENT.log_metrics(dic=metrics)
        return
    
    #def on_epoc_begin(self, epoc, logs=None):
    #    print('55555555 epoc begin 5555555')
    #    return
    
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        valLoss = logs.get('val_loss')
        #print('0000000 keras-auc: ', logs.get('val_auc'))
        #valLoss = 0.0
        y_true, y_pred = PPITrainTestCls.doEvaluation(self.model, self.valX, self.valY)
        
        try:
            currentAuc = PPILossCls.logTrainingMetrics(epoch, loss, valLoss, y_true, y_pred)
        except Exception as e: #in case of problems regarding gradient explosion (NaNs, Infs) we stop the training.
            self.model.stop_training = True
            logger.warn("### Stopping the training because of an exception (Nan/Inf)." + "\nException was: " + str(e))
            return
            
        if TrainingParams.MODEL_CHECK_POINT_MODE == -1:
            currentAuc = valLoss
        if TrainingParams.MODEL_CHECK_POINT_MODE * (currentAuc - self.bestAuc) > self.minDelta:
            #print('============== BestAuc changed: ', currentAuc, ' diff: ', currentAuc - self.bestAuc)
            self.bestAuc = currentAuc
            self.wait = 0
            self.bestWeights = self.model.get_weights() # Record the best weights if current results is better (more).
            self.bestEpoch = epoch
        elif TrainingParams.USE_EARLY_STOPPING == True:
            self.wait += 1
            if self.wait >= self.patience:
                self.stoppedEpoch = epoch
                self.model.stop_training = True
                #print('Epoch %05d: early stopping' % (self.stoppedEpoch + 1))
        return
    
    def on_train_end(self, logs=None):
        #print('Restoring model weights from the best epoch.')
        logger.info("Restoring model weights from the best epoch: {:02d} with the best VAL_AUC: {:.2%}".format(self.bestEpoch,self.bestAuc))
        self.model.set_weights(self.bestWeights)
        self.model.save(TrainingParams.MODEL_SAVE_FILE)
        return

class AttDataGenerator(Sequence):
    def __init__(self, model, features, labels):
        self.features = features
        self.labels = labels
        self.model = model
        self.epoc = 0
        # indicates number of batches called till now, across all epocs
        self.batchNum = 0   
        batchesPerEpoch = math.ceil(labels.shape[0] / TrainingParams.BATCH_SIZE)
        # indicates total number of bathes, across all epocs
        self.totalBatches = batchesPerEpoch * TrainingParams.NUM_EPOCS
        self.sampling = [True, False]
        #self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        sampleSize = self.features[0].shape[0]
        batchesPerEpoc = math.ceil(sampleSize / TrainingParams.BATCH_SIZE)
        return batchesPerEpoc

    def __getitem__(self, batchIndx):
        'Generate one batch of data'
        startIndx = batchIndx * TrainingParams.BATCH_SIZE
        endIndx = startIndx + TrainingParams.BATCH_SIZE
        origEncoderInput = self.features[0][startIndx:endIndx]
        origDecodeInput = self.features[1][startIndx:endIndx]
        newDecoderInput = self.getNewDecoderInput(origEncoderInput, origDecodeInput)
        newFeatures = [origEncoderInput, newDecoderInput]
        newLabels = self.labels[startIndx:endIndx]
        #print('startIndx: ', startIndx, ' | endIndx: ', endIndx)
        return newFeatures, newLabels

    def on_epoch_end(self):
        #print("===== epoc-end: ", self.epoc)
        self.epoc = self.epoc + 1
        return
    
    '''
        There are two sampling baskets: TRUE samples & MODEL prediction samples. We use 'scheduled sampling' to choose one or
        the other basket.
    '''
    def chooseSamplingBasket(self, h):
        def getUniformProb():
            sp, notsp = 0.50, 0.50
            return sp, notsp
        
        def getInvSigProb():
            i = h + self.batchNum
            k = self.totalBatches
            # calculate sampling probability based on this formula: p = k / (k + math.exp(i / k))
            sp = k / (k + math.exp(i/k))
            notsp = 1 - sp
            return sp, notsp
        
        #sp, notsp = getUniformProb()
        sp, notsp = getInvSigProb()
        #print('p =', sp, ' | 1-p=', notsp)
        samplingBasket = np.random.choice(self.sampling, p=[sp,notsp])
        #print('basket =', samplingBasket)
        return samplingBasket
    
    def getNewDecoderInput(self, origEncoderInput, origDecoderInput):
        self.batchNum = self.batchNum + 1
        modelInput = [origEncoderInput, origDecoderInput]
        _, y_pred = PPITrainTestCls.doEvaluation(self.model, modelInput, self.labels)
        # decoder.shape:  (16, 13, 32)  | y_pred.shape:  (16, 13, 32)
        #print("decoder.shape: ", origDecoderInput.shape, ' | y_pred.shape: ', y_pred.shape)
        newDecoderInput = np.copy(origDecoderInput)
        nd,hd,cd = y_pred.shape
        for n in range(nd):
            for h in range(hd):
                samplingBasket = self.chooseSamplingBasket(h)
                if samplingBasket == False:
                    newDecoderInput[n,h] = np.zeros(cd, DatasetParams.FLOAT_TYPE)
                    predClassId = np.argmax(y_pred[n,h])
                    newDecoderInput[n,h,predClassId] = 1
        return newDecoderInput
    
class PPITrainTestCls(object):
    @classmethod
    def setLogger(cls, loggerParam):
        global logger
        logger = loggerParam
    
    @classmethod
    def compileModel(cls, model):
        theLoss = PPILossCls.ppiLoss
        # syntax problem with 2.1; not working with 2.3.
        #aucMetric = tf.keras.metrics.AUC(num_thresholds=1000, curve='ROC', summation_method='interpolation', name='val_auc',
        #                                 dtype=DatasetParams.FLOAT_TYPE, thresholds=None, multi_label=False, label_weights=None
        #                                 )
        #theMetrics = [aucMetric]
        theMetrics = []
        if TrainingParams.OPT_FUN == TrainingParams.ADAM_OPT:
            # use  clipnorm or  clipvalue.
            # clipnorm=1.0 means: if the vector norm for a gradient exceeds 1.0, then the values in the vector will be rescaled so that the norm of 
            # the vector equals 1.0.
            # clipvalue=0.5 means: if a gradient value was less than -0.5, it is set to -0.5 and if it is more than 0.5, then it 
            # will be set to 0.5.
            theOptimizer = Adam(learning_rate=TrainingParams.LEARNING_RATE, beta_1=TrainingParams.DECAY_RATE1, beta_2=TrainingParams.DECAY_RATE2, 
                                #clipvalue=0.5,
                                #clipnorm=1.0,
                                epsilon=TrainingParams.EPSILON_RATE)
        elif TrainingParams.OPT_FUN == TrainingParams.SGD_OPT:
            #theOptimizer = SGD(TrainingParams.LEARNING_RATE)
            pass
        elif TrainingParams.OPT_FUN == TrainingParams.RMS_OPT:
            theOptimizer = RMSprop(learning_rate=TrainingParams.LEARNING_RATE, rho=0.9,  momentum=TrainingParams.MOMENTUM_RATE, 
                                   epsilon=TrainingParams.EPSILON_RATE)
        
        # we get problems (in custom-loss) if we don't add 'experimental_run_tf_function=False'.    
        #model.compile(loss=theLoss, optimizer=theOptimizer, metrics=theMetrics, experimental_run_tf_function=False)
        model.compile(loss=theLoss, optimizer=theOptimizer, metrics=theMetrics)
    
        return model
    
    @classmethod
    def getClassWeights(cls, trainingLabels):
        classWeights = dict(pd.Series(K.flatten(trainingLabels)).value_counts())
        # Number of 0's in the target set
        zeroClassCount = classWeights[0]
        #print('zzzzzzzz: ', zeroClassCount)
        # Number of 1's in the target set
        oneClassCount = classWeights[1]
        #print('oooooooo: ', oneClassCount)
        # In case number of 0's is much greater than number of 1's (we have imbalance data) then increase (by weighting) the importance of 1's. 
        classWeights = {0:1., 1:(zeroClassCount / oneClassCount)} 
        #classWeights = {0:1.00, 1:20.00} 
        print(classWeights)
        
        return classWeights
    
    @classmethod
    def getClassWeights2(cls, trainingLabels):
        classWeights = {0:1.00, 1:20.00} 
        #classWeights = [0.10, 0.90]     # doesn't have any effect.
        #print(classWeights)
        
        return classWeights
    
    @classmethod
    def doEvaluation(cls, model, valX, valY):
        def concatListItems(slicesList):
            concatenatedSlices = slicesList[0]
            for s in range(1, len(slicesList)):     
                slices = slicesList[s]
                concatenatedSlices = np.concatenate([concatenatedSlices, slices], axis=1)
            return concatenatedSlices
        
        #print('len(valx): ', len(valX))
        #print('valx[].shape: ', valX[10].shape)
        if DatasetParams.USE_VAR_BATCH_INPUT:
            # valX = [array(1, 159, 41), ..., array(1, 88, 41), ..., array(1, 80, 41)] 
            #print('valx[].shape: ', valX[10].shape)
            #print('valx[]: ', valX[10])
            #print('valy[]: ', valY[10])
            trueList, predList = [], []
            for i in range(len(valX)):
                trueList.append(valY[i])
                #predList.append(model.predict(valX[i], batch_size=TrainingParams.BATCH_SIZE)) 
                predList.append(model.predict(valX[i], batch_size=None)) 
                #print(trueList[i].shape)
                #print(predList[i].shape)
            y_true = concatListItems(trueList)    
            y_pred = concatListItems(predList) 
        else:
            y_true = valY
            #y_pred = model.predict(valX, verbose=TrainingParams.VERBOSE_INDICATOR, batch_size=TrainingParams.BATCH_SIZE)
            y_pred = model.predict(valX, verbose=TrainingParams.VERBOSE_INDICATOR, batch_size=None)
        
        #y_true.shape:  (24, 512, 1) ;; y_pred.shape:  (24, 512, 1)
        #y_true.shape:  (1, 4417, 1) ;; y_pred.shape:  (1, 4417, 1)
        #print('y_true.shape: ', y_true.shape)
        #print('y_pred.shape: ', y_pred.shape)
        
        return y_true, y_pred
    
    @classmethod
    def doAttEvaluation(cls, valX, valY):
        def beamSearch(candSeqs):
            #print('@@@@@@ length candSeqs: ', len(candSeqs))
            #a = 0.7
            #ty = DatasetParams.ATT_DECODER_INPUT_LEN ** a
            # candSeqs.shape: [ [(class-id, ATT_LABEL_DIM), ..., (class-id, ATT_LABEL_DIM)] .. [(class-id, ATT_LABEL_DIM), ..., (class-id, ATT_LABEL_DIM)] ] 
            prevLogProb = -float('inf') # because log of probs are negative numbers.
            preferredSeq = []
            for candSeq in candSeqs:
                curLogProb = 0
                for t in candSeq:
                    predClassId, predProbs = t
                    predProb = predProbs[predClassId]
                    curLogProb = curLogProb + np.log(predProb)
                # length normalization
                #curLogProb = curLogProb / ty
                if curLogProb >= prevLogProb:
                    prevLogProb = curLogProb
                    preferredSeq = candSeq  
             
            exam_y_pred = np.zeros(tmpLabelShape, DatasetParams.FLOAT_TYPE)
            for t in range(len(preferredSeq)):
                predClassId, predProbs = preferredSeq[t]
                predProbs[predClassId] = 1.0 
                exam_y_pred[t] = predProbs
            return exam_y_pred
        
        def consNextDecIns(decOut):
            # Get indexes of the #beam-width highest probabilities predicted. Note that the result is sorted from highest to less high.
            predClassIds = hq.nlargest(numHighestProbs, range(len(decOut[0,0])), decOut[0,0].take)
            for b in range(TrainingParams.ATT_BEAM_WIDTH):
                tmpDecIn = np.zeros(decInShape, DatasetParams.FLOAT_TYPE)
                predClassId = predClassIds[b]
                tmpDecIn[:,:,predClassId] = 1
                if np.array_equal(tmpDecIn[0,0], DatasetParams.ATT_DECODER_START) or np.array_equal(tmpDecIn[0,0], DatasetParams.ATT_DECODER_END):
                    tmpDecIn[:,:,predClassId] = 0
                    #predClassId = np.random.randint(numRealClasses)
                    predClassId = predClassIds[-2]
                    tmpDecIn[:,:,predClassId] = 1
                    if np.array_equal(tmpDecIn[0,0], DatasetParams.ATT_DECODER_START) or np.array_equal(tmpDecIn[0,0], DatasetParams.ATT_DECODER_END):
                        tmpDecIn[:,:,predClassId] = 0
                        #predClassId = np.random.randint(numRealClasses)
                        predClassId = predClassIds[-1]
                        tmpDecIn[:,:,predClassId] = 1
                #print('tmpDecIn: ',tmpDecIn.shape,tmpDecIn)
                predTree.create_node(parent=leaveNode.identifier, data=(predClassId,decOut[0,0],tmpDecIn))
                
                if False:
                    trueClass = y_true[n,h]
                    trueClassId = np.argmax(trueClass)
                    if (trueClassId == predClassId):
                        print("trueClassId: ", trueClassId) 
                        print("predClassId: ", predClassId) 
            return
        
        def consCandSeqs():
            # Get a list of all pathes in the tree. Each path is a list of ids of a node.
            pleaves = predTree.paths_to_leaves()    
            candSeqs = []
            for pleave in pleaves:
                candSeq = []
                pleave.pop(0)
                for ni in pleave:
                    predClassId,predProbs,_ = predTree.get_node(ni).data
                    candSeq.append((predClassId,predProbs))
                candSeqs.append(candSeq)
            return candSeqs
        
        numHighestProbs = TrainingParams.ATT_BEAM_WIDTH + 2  # additional one for replacing start/end token if wrongly predicted.
        tmpLabelShape = (DatasetParams.ATT_DECODER_INPUT_LEN, DatasetParams.ATT_LABEL_DIM)   #(97, 128)
        #numRealClasses = DatasetParams.ATT_LABEL_DIM // 2
        
        encoder_model, decoder_model = TrainingParams.ATT_ENCODER_MODEL, TrainingParams.ATT_DECODEL_MODEL
        #print(valX.shape, valY.shape) #(437, 576, 44) (437, 98, 128)
        #print('start.shape: ', DatasetParams.ATT_DECODER_START.shape)   #(128,)
        nd,_,_ = valX.shape
        y_true = valY[:, 1:, :]
        labelShape = (nd, DatasetParams.ATT_DECODER_INPUT_LEN, DatasetParams.ATT_LABEL_DIM)   #(437, 97, 128) 
        y_pred = np.zeros(labelShape, DatasetParams.FLOAT_TYPE)
        decInShape = (1, 1, DatasetParams.ATT_LABEL_DIM)
        initDecIn = np.reshape(DatasetParams.ATT_DECODER_START, decInShape)
        for n in range(nd):
            if TrainingParams.USE_BIDIRECTIONAL_ATT:
                enc_outs, enc_fwd_state, enc_back_state = encoder_model.predict(valX[n:n+1])
                dec_fwd_state, dec_back_state = enc_fwd_state, enc_back_state
                #dec_state = np.concatenate([enc_fwd_state, enc_back_state], axis=-1)
            else:
                #enc_outs, dec_state = encoder_model.predict(valX[n:n+1])
                dec_state = encoder_model.predict(valX[n:n+1])
            
            # Create one tree per example. Data of each node is a tuple: (class-id,class-vector-containing-probs,class-vector-containing-one-hot).
            # We need the second item to calculate the highest probabilities of a path in the tree. 
            # The third one provides input for the decoder model.
            predTree = Tree()   
            predTree.create_node(parent=None, data=(None,None,initDecIn))
            # (1, 576, 128) (1, 64) (1, 64) (1, 128) (1,1,128)=
            #print(enc_outs.shape, enc_fwd_state.shape, enc_back_state.shape, decIn.shape)
            for h in range(DatasetParams.ATT_DECODER_INPUT_LEN):
                leaveNodes = predTree.leaves()  # return a list of all leave nodes
                for leaveNode in leaveNodes:
                    _,_,decIn = leaveNode.data
                    if TrainingParams.USE_BIDIRECTIONAL_ATT:
                        decOut, dec_fwd_state, dec_back_state = decoder_model.predict([enc_outs, dec_fwd_state, dec_back_state, decIn])
                        #decOut, dec_state = decoder_model.predict([enc_outs, dec_state, decIn])
                    else:
                        #decOut, dec_state = decoder_model.predict([enc_outs, dec_state, decIn])
                        decOut, dec_state = decoder_model.predict([decIn, dec_state])
                    #print("=======" ,decOut.shape) #(1,1,128)
                    #print('decOut: ', decOut)  
                    consNextDecIns(decOut)
            candSeqs = consCandSeqs()
            y_pred[n] = beamSearch(candSeqs)
            
        return y_true, y_pred
    
    @classmethod
    def conv2AttInputs(cls, inputX, inputY):
        # inputY=[START...END] and decoderInput=[START...]
        decoderInput = inputY[:, :-1, :]
        inputX = [inputX, decoderInput]
        
        # inputY[0,5] means: first protein 6e grid (6 because the first grid is the START indicator).
        #print(inputY[0,1])
        #print(inputY[0,3])
        #print(inputY[0,-1])
        #print(decoderInput[0,6])
        #print(decoderInput[0,0])
        
        # inputY=[...END] without START
        inputY = inputY[:, 1:, :]
        return inputX, inputY
    
    @classmethod
    def doTraining(cls, model, trainX, valX, trainY, valY):
        def detStepsPerEpoc(features):
            sampleSize = len(features)
            stepsPerEpoc = 0
            for i in range(sampleSize):
                feature = features[i]
                numBatches = feature.shape[0] // TrainingParams.BATCH_SIZE
                rem = feature.shape[0] % TrainingParams.BATCH_SIZE
                if rem != 0:
                    numBatches = numBatches + 1
                stepsPerEpoc = stepsPerEpoc + numBatches
            #print('========== stepsPerEpoc: ', stepsPerEpoc)
            return stepsPerEpoc
        
        def varLenFitGenerator2(features, labels):
            sampleSize = len(features)
            #print('samle size: ', sampleSize)
            while True:
                for i in range(0, sampleSize):
                    feature, label = features[i], labels[i]
                    #print(feature.shape)
                    yield feature, label
                
        def varLenFitGenerator(features, labels):
            while True:
                # features = [(58, 64, 41), ..., (15, 1024, 41)] || labels = [(58, 64, 1), ..., (15, 1024, 1)] or
                # features = [(1, 159, 41), ..., (1, 88, 41)] || labels = [(1, 159, 1), ..., (1, 88, 1)] 
                sampleSize = len(features)
                #print('========== sampleSize: ', sampleSize)
                for i in range(sampleSize):
                    feature, label = features[i], labels[i]
                    numBatches = feature.shape[0] // TrainingParams.BATCH_SIZE  #58/8 or 1/8
                    rem = feature.shape[0] % TrainingParams.BATCH_SIZE
                    if rem != 0:
                        numBatches = numBatches + 1
                    #print('========== numBatches: ',numBatches)
                    fpart = np.array_split(feature, numBatches)
                    lpart = np.array_split(label, numBatches)
                    for j in range(len(fpart)):
                        #print('========== j: ',j)
                        yield fpart[j], lpart[j]

        def lrScheduler(epoch):
            if epoch < 10:
                return TrainingParams.LEARNING_RATE
            else:
                return TrainingParams.LEARNING_RATE * np.exp(0.1 * (10 - epoch))
        
        mcAtMaxAucCallBack = ModelCheckPointAtMaxAuc()
        callBackOps = [mcAtMaxAucCallBack]

        if TrainingParams.USE_REDUCE_LR: # we can't use it because of a bug.
            lrCallBack = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            callBackOps.append(lrCallBack)
        
        if TrainingParams.USE_LR_SCHEDULER: # we can't use it because of a bug.
            lrsCallBack = LearningRateScheduler(lrScheduler)
            callBackOps.append(lrsCallBack)
            
        if TrainingParams.USE_TENSOR_BOARD:    
            # Not supported in eager mode. 
            tbCallBack = TensorBoard(log_dir=TrainingParams.TB_STAT_DIR, histogram_freq=5, write_graph=True, write_grads=True)
            callBackOps.append(tbCallBack)
            
        if TrainingParams.USE_CLASS_WEIGHT:
            #classWeight = cls.getClassWeights(np.concatenate([trainY, valY]))
            classWeight = cls.getClassWeights(trainY)
        else:
            classWeight = None    
        
        if DatasetParams.USE_ATT:
            trainX, trainY = cls.conv2AttInputs(trainX, trainY)
            valX, valY = cls.conv2AttInputs(valX, valY)
        
        mcAtMaxAucCallBack.setValset(valX, valY)   
        trainingStartTime = time.time()
        # You can't use USE_VAR_BATCH_INPUT together with seq2seq (USE_ATT).
        if DatasetParams.USE_VAR_BATCH_INPUT:
            #print('trainX[].shape', trainX[0].shape)
            #print('trainX[]', trainX[0])
            #print('trainY[]', trainY[0])
            #model.fit_generator(generator=varLenFitGenerator(trainX, trainY), steps_per_epoch=len(trainX),
                                #epochs=TrainingParams.NUM_EPOCS, shuffle=not LossParams.USE_WEIGHTED_LOSS,
                                #epochs=TrainingParams.NUM_EPOCS, shuffle=False,
                                #validation_data=varLenFitGenerator(valX, valY), validation_steps=len(valX),
                                #callbacks=callBackOps, verbose=TrainingParams.VERBOSE_INDICATOR, class_weight=classWeight)
            trainGen = varLenFitGenerator(trainX, trainY)
            valGen = varLenFitGenerator(valX, valY)
            model.fit(x=trainGen, 
                      #steps_per_epoch=len(trainX),
                      steps_per_epoch=detStepsPerEpoc(trainX),
                      epochs=TrainingParams.NUM_EPOCS,
                      shuffle=not LossParams.USE_WEIGHTED_LOSS,
                      validation_data=valGen, validation_steps=detStepsPerEpoc(valX),
                      callbacks=callBackOps, verbose=TrainingParams.VERBOSE_INDICATOR, class_weight=classWeight)
        elif False and DatasetParams.USE_ATT:
            attDataGen = AttDataGenerator(model, trainX, trainY)
            model.fit_generator(generator=attDataGen, 
                                #epochs=TrainingParams.NUM_EPOCS, shuffle=not LossParams.USE_WEIGHTED_LOSS,
                                epochs=TrainingParams.NUM_EPOCS, shuffle=False,
                                validation_data=(valX, valY), 
                                #use_multiprocessing=True, workers=6, #is some kind of bug
                                callbacks=callBackOps, verbose=TrainingParams.VERBOSE_INDICATOR, class_weight=classWeight)
        
        else:
            #print('type(trainX)', type(trainX))
            #print('type(trainX[])', type(trainX[0]))
            #print('trainX[].shape', trainX[0].shape)
            #print('len(valx): ', len(valX))
            model.fit(x=trainX, y=trainY, epochs=TrainingParams.NUM_EPOCS,
                      batch_size=TrainingParams.BATCH_SIZE, shuffle=not LossParams.USE_WEIGHTED_LOSS,
                      validation_data=(valX, valY),
                      callbacks=callBackOps, verbose=TrainingParams.VERBOSE_INDICATOR, class_weight=classWeight)
        trainingEndTime = time.time()
        TrainingParams.TRAIN_DURATION = str(datetime.timedelta(seconds=trainingEndTime-trainingStartTime))
        logger.info("Training time: {}".format(TrainingParams.TRAIN_DURATION))
     
        return 
    
    @classmethod
    def logTestExperiment(cls):
        if DatasetParams.COMET_EXPERIMENT is not None:
            params = {
                "TEST_SHAPE": TrainingParams.TEST_SHAPE,
                "ENS_ARCHS": TrainingParams.ENSEMBLE_ARCHS,
                "NUM_ENS_ARCHS": len(TrainingParams.ENSEMBLE_ARCHS),
            }
            DatasetParams.COMET_EXPERIMENT.log_parameters(params)
        return
    
    @classmethod  
    def getMaxProtLen(cls):
        # The max protein length is PROT_IMAGE_H.
        if len(TrainingParams.INPUT_SHAPE) == 2:
            maxProtLen = TrainingParams.INPUT_SHAPE[0]
        # The max protein length is PROT_IMAGE_H * PROT_IMAGE_W if 2D-model used.
        elif len(TrainingParams.INPUT_SHAPE) == 3:
            maxProtLen = TrainingParams.INPUT_SHAPE[0] * TrainingParams.INPUT_SHAPE[1]
        else:
            raise Exception("Invalid input-shape: ", TrainingParams.INPUT_SHAPE)
            
        return maxProtLen
    
    @classmethod    
    def savePredFile(cls, dataFile, flat_y_true, flat_y_pred, testingLabel):
        testDataset = pd.read_csv(dataFile)
        protSeqs = testDataset.loc[:, DatasetParams.SEQ_COLUMN_NAME].values.astype(str)
        protIds = testDataset.loc[:, DatasetParams.PROT_ID_NAME].values
        protLens = testDataset.loc[:, DatasetParams.REAL_LEN_COL_NAME].values
        numProts = len(protSeqs)
        starti = 0
        predData = []
        flat_y_true = flat_y_true.numpy()
        flat_y_pred = flat_y_pred.numpy()
        for proti in range(0, numProts):
            protLen = protLens[proti]
            # If we have removed proteins longer than max-len then we have to remove them here as well.
            maxProtLen = cls.getMaxProtLen()
            if DatasetParams.SKIP_SLICING == True and protLen > maxProtLen:
                continue
            protId = protIds[proti]
            protAA = protSeqs[proti] #'str,...,str'
            prot_y_true = ",".join(str(e) for e in flat_y_true[starti:starti+protLen]) #'float,...,float'
            prot_y_pred = ",".join(str(e) for e in flat_y_pred[starti:starti+protLen]) #'float,...,float'
            starti = starti + protLen
            predData.append([protId,protAA,prot_y_true,prot_y_pred])
            
        #predFile = TrainingParams.MODEL_PRED_FILE.format(TrainingParams.ALGRITHM_NAME, testingLabel)
        predFile = TrainingParams.MODEL_PRED_FILE.format(testingLabel)
        dataset = pd.DataFrame(predData, columns=PPIPredPlotCls.PRED_COLUMNS)
        print("\n## Generating pred-data for @@: " + predFile + " @@")
        dataset.to_csv(predFile, index=False) 
        return 
    
    @classmethod    
    def saveMetricsPerProtFile(cls, numProts, protIds, protLens, all_y_trues, all_y_preds, testingLabel):
        def consProtMetricsData():
            SIMPLE,YOUDEN,OPTIMAL,EQUAL = 0,1,2,3   #index of different cutoff-methods.
            scoreDataDict = scoreDataDicts[EQUAL]
            prorMetricsData = []
            #note: the order of the followig satements is important.
            prorMetricsData.append(protIds[proti])
            prorMetricsData.append(protLen)
            prorMetricsData.append(curveDataDict['aucScore'])
            prorMetricsData.append(curveDataDict['apScore'])
            prorMetricsData.append(scoreDataDict['mcc'])
            prorMetricsData.append(scoreDataDict['accuracy'])
            prorMetricsData.append(scoreDataDict['precision'])
            prorMetricsData.append(scoreDataDict['specificity'])
            prorMetricsData.append(scoreDataDict['recall'])
            prorMetricsData.append(scoreDataDict['f1'])
            prorMetricsData.append(scoreDataDict['confusion[TN,FP,FN,TP]'])
            prorMetricsData.append(scoreDataDict['cutoff'])
            return prorMetricsData

        starti = 0
        allMetricsData = []
        for proti in range(0, numProts):
            protLen = protLens[proti]
            ##getMetrics expects 'list of flat_y_trues and y_preds'. So we must pack them in lists.
            prot_y_trues = [all_y_trues[starti:starti+protLen]]
            prot_y_preds = [all_y_preds[starti:starti+protLen]]
            scoreDataDicts, curveDataDict = PPIPredMericsCls.getMetrics(prot_y_trues, prot_y_preds)
            allMetricsData.append(consProtMetricsData())
            starti = starti + protLen
            
        metricsFile = TrainingParams.METRICS_PER_PROT_FILE.format(TrainingParams.ALGRITHM_NAME, testingLabel)
        dataset = pd.DataFrame(allMetricsData, columns=PPIPredPlotCls.MERICS_PER_PROT_COLUMNS)
        print("\n## Generating metrics-data per protein for @@: " + metricsFile + " @@")
        dataset.to_csv(metricsFile, index=False) 
        return
    
    @classmethod 
    def genPredPlotFromSavedFile(cls, testingLabel):
        #predFile = TrainingParams.MODEL_PRED_FILE.format(TrainingParams.ALGRITHM_NAME, testingLabel)
        predFile = TrainingParams.MODEL_PRED_FILE.format(testingLabel)
        figFile = PPIPredPlotCls.generatePredPlot(predFile, testingLabel, TrainingParams.ALGRITHM_NAME)
        return figFile
    
    @classmethod
    def getEnsDataset(cls, training, testDatasetIndex=None):
        if training:
            datasetLabel = DatasetParams.EXPR_TRAINING_LABEL
            #ensDataFile = TrainingParams.ENS_TRAIN_FILE.format(TrainingParams.ALGRITHM_NAME, datasetLabel)
            ensDataFile = TrainingParams.ENS_TRAIN_FILE.format(TrainingParams.ALGRITHM_NAME, datasetLabel)
        else:
            datasetLabel = DatasetParams.EXPR_TESTING_LABELS[testDatasetIndex]
            #ensDataFile = TrainingParams.ENS_TEST_FILE.format(TrainingParams.ALGRITHM_NAME, datasetLabel)
            ensDataFile = TrainingParams.ENS_TEST_FILE.format(datasetLabel)
        
        ensDataset = pd.read_csv(ensDataFile) 
        ensDataset = ensDataset.to_numpy()          #(84941,7) 
        ensDataset = np.hsplit(ensDataset,[-1])     #[(84941,6),(84941,1)]
        inputData = np.reshape(ensDataset[0], (ensDataset[0].shape[0],1,ensDataset[0].shape[1]))    #(84941,1,6)
        labelData = np.reshape(ensDataset[1], (ensDataset[1].shape[0],1,1))                         #(84941,1,1)
        ensDataset = [inputData,labelData]          #[(84941,1,6),(84941,1,1)]
        return ensDataset
    
    @classmethod
    def saveEnsDataset(cls, ensDataset, training, testDatasetIndex=None):
        #ensDataset=[inputData,labelData]=[(84941,1,6),(84941,1,1)]
        inputData = np.reshape(ensDataset[0], (ensDataset[0].shape[0],ensDataset[0].shape[2]))  #(84941,6)
        labelData = np.reshape(ensDataset[1], (ensDataset[1].shape[0],ensDataset[1].shape[2]))  #(84941,1)
        ensData = np.concatenate([inputData,labelData], axis=1)
        ensArchs = TrainingParams.ENSEMBLE_ARCHS
        TrainingParams.ENS_DATA_COLUMNS = []
        for ea in ensArchs:
            TrainingParams.ENS_DATA_COLUMNS.append(ea)
        TrainingParams.ENS_DATA_COLUMNS.append('y_true')
        
        if training:
            datasetLabel = DatasetParams.EXPR_TRAINING_LABEL
            #ensDataFile = TrainingParams.ENS_TRAIN_FILE.format(TrainingParams.ALGRITHM_NAME, datasetLabel)
            ensDataFile = TrainingParams.ENS_TRAIN_FILE.format(datasetLabel)
        else:
            datasetLabel = DatasetParams.EXPR_TESTING_LABELS[testDatasetIndex] 
            #ensDataFile = TrainingParams.ENS_TEST_FILE.format(TrainingParams.ALGRITHM_NAME, datasetLabel)
            ensDataFile = TrainingParams.ENS_TEST_FILE.format(datasetLabel)
        dataset = pd.DataFrame(ensData, columns=TrainingParams.ENS_DATA_COLUMNS)
        print("\n## Generating ensemble-data for @@: " + ensDataFile + " @@")
        dataset.to_csv(ensDataFile, index=False)
        return
    
    @classmethod    
    def genPredPlotFromMemory(cls, dataFile, flat_y_true, flat_y_pred, testingLabel):
        testDataset = pd.read_csv(dataFile)
        aavec = testDataset.loc[:, DatasetParams.SEQ_COLUMN_NAME].values.astype(str)
        protIds = testDataset.loc[:, DatasetParams.PROT_ID_NAME].values
        protLens = testDataset.loc[:, DatasetParams.REAL_LEN_COL_NAME].values
        numProts = len(aavec)
        
        if TrainingParams.GEN_METRICS_PER_PROT:
            cls.saveMetricsPerProtFile(numProts, protIds, protLens, flat_y_true, flat_y_pred, testingLabel)
        
        starti = 0
        y_trues = []
        y_preds = []
        protSeqs = []
        for proti in range(0, numProts):
            protLen = protLens[proti]
            # If we have removed proteins longer than max-len then we have to remove them here as well.
            maxProtLen = cls.getMaxProtLen()
            if DatasetParams.SKIP_SLICING == True and protLen > maxProtLen:
                continue
            protAA = aavec[proti].split(',')
            protSeqs.append(protAA)
            y_trues.append(flat_y_true[starti:starti+protLen])
            y_preds.append(flat_y_pred[starti:starti+protLen])
            starti = starti + protLen
        
        #predFile = TrainingParams.MODEL_PRED_FILE.format(TrainingParams.ALGRITHM_NAME, testingLabel)
        predFile = TrainingParams.MODEL_PRED_FILE.format(testingLabel)
        figFile = PPIPredPlotCls.plotProtPreds(predFile, protIds, protSeqs, y_trues, y_preds, testingLabel, TrainingParams.ALGRITHM_NAME)
        return figFile
    
    @classmethod
    def plotMetrics(cls, testingFile, flat_y_true, flat_y_pred, testingLabel):
        if TrainingParams.SAVE_PRED_FILE:
            cls.savePredFile(testingFile, flat_y_true, flat_y_pred, testingLabel)
            figFile = cls.genPredPlotFromSavedFile(testingLabel)
        else:
            figFile = cls.genPredPlotFromMemory(testingFile, flat_y_true, flat_y_pred, testingLabel)
        
        if DatasetParams.COMET_EXPERIMENT is not None:
            DatasetParams.COMET_EXPERIMENT.log_asset(figFile)
        
        return 
    
    @classmethod
    def plotStackedMetrics(cls, modelsNames, modelsResults, ensResults):
        #modelsResults=[[(99318,1,6), (99318,1,1)],    [(64609,1,6), (64609,1,1)]] #list of list of 3-D arrays; one list per test data set
        #ensResults=   [[(99318,),    (99318,)   ],    [(64609,),    (64609,)]]    #list of list of 1-D arrays = [preds,trues]
        for i in range(len(DatasetParams.EXPR_TESTING_FILE_SET)):
            modelsResult = modelsResults[i]
            ensResult = ensResults[i]  #[preds,trues]
            ensPreds = ensResult[0]
            y_stacked_preds = modelsResult[0]
            #add ensPreds to the stack
            ensPreds = np.reshape(ensPreds, (ensPreds.shape[0],1))
            y_stacked_preds = np.dstack([y_stacked_preds,ensPreds])
            logger.info('=======y_stacked_preds: ' + str(y_stacked_preds.shape))
            y_trues = ensResult[1]   # this is the same for all models, including ensnet.
            testingLabel = DatasetParams.EXPR_TESTING_LABELS[i]
            #predFile = TrainingParams.MODEL_PRED_FILE.format(TrainingParams.ALGRITHM_NAME, testingLabel)
            predFile = TrainingParams.MODEL_PRED_FILE.format(testingLabel)
            figFile = PPIPredPlotCls.plotStackedProtPreds(predFile, y_trues, y_stacked_preds, modelsNames, testingLabel)
            
            if DatasetParams.COMET_EXPERIMENT is not None:
                DatasetParams.COMET_EXPERIMENT.log_asset(figFile)
        return
    
    @classmethod
    def plotMetricsPerProt(cls, modelsAlgs, modelsNames, plotType):
        for i in range(len(DatasetParams.EXPR_TESTING_FILE_SET)):
            testingLabel = DatasetParams.EXPR_TESTING_LABELS[i]
            metricsPerModel = []
            for alg in modelsAlgs:
                modelMetricsFile = TrainingParams.METRICS_PER_PROT_FILE.format(alg, testingLabel)
                modelMetricsDF = pd.read_csv(modelMetricsFile)
                metricsPerModel.append(modelMetricsDF)
            ensMetricsFile = TrainingParams.METRICS_PER_PROT_FILE.format(TrainingParams.ALGRITHM_NAME, testingLabel)
            figFile = PPIPredPlotCls.plotCombinedMetricsPerProt(ensMetricsFile, metricsPerModel, modelsNames, testingLabel, plotType)
            
            if DatasetParams.COMET_EXPERIMENT is not None:
                DatasetParams.COMET_EXPERIMENT.log_asset(figFile)
        return
    
    @classmethod
    def doKerasModelTesting(cls, model, testX, testY):
        if True and DatasetParams.USE_ATT:
            y_true, y_pred = cls.doAttEvaluation(testX, testY)
        else:
            if False and DatasetParams.USE_ATT:
                testX, testY = cls.conv2AttInputs(testX, testY)
            y_true, y_pred = cls.doEvaluation(model, testX, testY)
        return y_true, y_pred 
    
    @classmethod
    def doEnsXGBModelTesting(cls, model, testX, testY):
        NUM_MODELS = testX.shape[2]
        testX = np.reshape(testX, (testX.shape[0], NUM_MODELS))   #(82663,5)
        y_true = testY.flatten()   
        y_pred = model.predict(testX)            #if the model trained with RFRegressor, which returns (per-sample,) = (82663,)
        return y_true, y_pred 
    
    @classmethod
    def doEnsRFModelTesting(cls, model, testX, testY):
        NUM_MODELS = testX.shape[2]
        testX = np.reshape(testX, (testX.shape[0], NUM_MODELS))   #(82663,5)
        y_true = testY.flatten()   
        y_pred = model.predict(testX)            #if the model trained with RFRegressor, which returns (per-sample,) = (82663,)
        #y_pred = model.predict_proba(testX)[:,1]  #if the model trained with RFClassifier, which returns (per-sample, per-class) = (82663,2)
        return y_true, y_pred 
    
    @classmethod
    def doEnsDirModelTesting(cls, bestDirWeights, testX, testY):
        NUM_MODELS = testX.shape[2]
        y_true = testY.flatten()   
        #multiply the bestDirWeights[0] to the predictions of all examples of models[0], ...
        y_pred = np.array([testX[:,:,i].flatten() * bestDirWeights[i] for i in range(NUM_MODELS)])    #(5, 82663)
        #sum weighted predictions of all models for each example
        y_pred = np.sum(y_pred, axis=0)   #(82663,)
        return y_true, y_pred 
    
    @classmethod
    def doTesting(cls, model, testingFile, testingLabel, testDataset=None):
        testingDateTime = datetime.datetime.now().strftime("%d-%m-%Y#%H:%M:%S")
        logger.info("## Testing " + testingFile + " at: " + str(testingDateTime) + " ##")
        #testX.shape: (58,1024,28) & testY.shape: (58,1024,1)
        if testDataset is None:
            _, testX, _, testY = PPIDatasetCls.makeDataset(testingFile, TrainingParams.INPUT_SHAPE, 
                                                           TrainingParams.LABEL_SHAPE, False, TrainingParams.USE_DATASET_PARTITIONING)
        else:   #this is for ensemble testing where the inputs are predictions of individual architecture
            testX = testDataset[0]  #(82663, 1, 5)
            testY = testDataset[1]  #(82663, 1, 1)
            
        if not DatasetParams.USE_VAR_BATCH_INPUT:
            TrainingParams.TEST_SHAPE.append([testingLabel, testX.shape, testY.shape])
        
        testingStartTime = time.time()
        if TrainingParams.ENS_DIRICHLET_MODEL:  #almost the same performance as the Keras ANN
            y_true, y_pred = cls.doEnsDirModelTesting(model, testX, testY)  
        elif TrainingParams.ENS_RF_MODEL:       #performance is slightly worse: RFRegressor > RFClassifier
            y_true, y_pred = cls.doEnsRFModelTesting(model, testX, testY)  
        elif TrainingParams.ENS_XGBOOST_MODEL:       
            y_true, y_pred = cls.doEnsXGBModelTesting(model, testX, testY) 
        else:
            y_true, y_pred = cls.doKerasModelTesting(model, testX, testY)  
            
        testingEndTime = time.time()
        logger.info("Testing time: {}".format(datetime.timedelta(seconds=testingEndTime-testingStartTime)))
        
        if TrainingParams.USE_ENSEMBLE_TRAINING or TrainingParams.USE_ENSEMBLE_TESTING:
            return y_true, y_pred
        
        flat_y_true, flat_y_pred = PPILossCls.logTestingMetrics(y_true, y_pred, testingLabel)
        cls.plotMetrics(testingFile, flat_y_true, flat_y_pred, testingLabel)    
        
        return flat_y_true, flat_y_pred
    
    @classmethod
    def testModelForEnsembl(cls, testDatasetIndex):
        model = load_model(TrainingParams.MODEL_SAVE_FILE, compile=False, custom_objects=TrainingParams.CUSTOM_OBJECTS)
        
        if TrainingParams.USE_ENSEMBLE_TRAINING: #use validation dataset of the training dataset if we perform ensemble testing
            y_true, y_pred = cls.doTesting(model, DatasetParams.EXPR_TRAINING_FILE, DatasetParams.EXPR_TRAINING_LABEL)
        else:
            y_true, y_pred = cls.doTesting(model, DatasetParams.EXPR_TESTING_FILE_SET[testDatasetIndex], DatasetParams.EXPR_TESTING_LABELS[testDatasetIndex])
        
        return y_true,y_pred
    
    @classmethod
    def testModel(cls, testDatasets=None):
        print("%%%%%%% loading the saved model: ", TrainingParams.MODEL_SAVE_FILE)
        if TrainingParams.ENS_DIRICHLET_MODEL or TrainingParams.ENS_RF_MODEL or TrainingParams.ENS_XGBOOST_MODEL:
            model = joblib.load(TrainingParams.MODEL_SAVE_FILE)
        else:
            #model = load_model(TrainingParams.MODEL_SAVE_FILE, compile=True)
            #model = load_model(TrainingParams.MODEL_SAVE_FILE, compile=False)
            model = load_model(TrainingParams.MODEL_SAVE_FILE, compile=False, custom_objects=TrainingParams.CUSTOM_OBJECTS)
        
        tstResults = []
        for i in range(len(DatasetParams.EXPR_TESTING_FILE_SET)):
            if testDatasets is None:
                y_true, y_pred = cls.doTesting(model, DatasetParams.EXPR_TESTING_FILE_SET[i], DatasetParams.EXPR_TESTING_LABELS[i])
            else:
                y_true, y_pred = cls.doTesting(model, DatasetParams.EXPR_TESTING_FILE_SET[i], DatasetParams.EXPR_TESTING_LABELS[i], testDatasets[i])
            tstResult = [y_pred, y_true]
            tstResults.append(tstResult)
        
        cls.logTestExperiment()
        
        return tstResults
    
    @classmethod
    def prepareEnsembledPreds(cls, stacked_y_true, stacked_y_pred, cur_y_true, cur_y_pred):
        # stacked_y_true = (64*1024,) of one algorithm (e.g., rnn-ppi) -- they are flattened.
        # cur_y_true = (64,1024,1) of one algorithm (e.g., rnn-ppi)
        if stacked_y_true is None:
            cur_y_true, cur_y_pred = PPILossCls.maskPadTargetTensor(cur_y_true, cur_y_pred)
            cur_y_true = np.reshape(cur_y_true, (cur_y_true.shape[0],1,1))
            cur_y_pred = np.reshape(cur_y_pred, (cur_y_pred.shape[0],1))
            return cur_y_true, cur_y_pred
        
        _, cur_y_pred = PPILossCls.maskPadTargetTensor(cur_y_true, cur_y_pred)
        cur_y_pred = np.reshape(cur_y_pred, (cur_y_pred.shape[0],1))
        stacked_y_pred = np.dstack([stacked_y_pred,cur_y_pred])
            
        return stacked_y_true, stacked_y_pred
    
    @classmethod
    def printModel(cls, model):
        model.summary(print_fn=logger.info)
        plot_model(model, to_file=TrainingParams.MODEL_PLOT_FILE, show_shapes=True, show_layer_names=True)
    
    @classmethod
    def logTrainExperiment(cls):
        if DatasetParams.COMET_EXPERIMENT is not None:
            DatasetParams.COMET_EXPERIMENT.log_model(TrainingParams.ALGRITHM_NAME, TrainingParams.MODEL_SAVE_FILE)
            DatasetParams.COMET_EXPERIMENT.log_image(TrainingParams.MODEL_PLOT_FILE, name='Model Graph')
            DatasetParams.COMET_EXPERIMENT.log_asset(TrainingParams.MODEL_PARAMS_FILE)
            params = {
                "OPT_FUN": TrainingParams.OPT_NAMES[TrainingParams.OPT_FUN],
                "TRAIN_SHAPE": TrainingParams.TRAIN_SHAPE,
                "VAL_SHAPE": TrainingParams.VAL_SHAPE,
                "KERNAL_INITIALIZER": TrainingParams.KERNAL_INITIALIZER.__name__,
                "ACTIVATION_FUN": TrainingParams.ACTIVATION_FUN.__name__,
                "USE_BIAS": TrainingParams.USE_BIAS,
                "DROPOUT_RATE": TrainingParams.DROPOUT_RATE,
                "USE_EARLY_STOPPING": TrainingParams.USE_EARLY_STOPPING,
                "MODEL_CHECK_POINT_MODE": TrainingParams.MODEL_CHECK_POINT_MODE,
                "FEATURE_COLUMNS": DatasetParams.FEATURE_COLUMNS,
                "FLOAT_TYPE": DatasetParams.FLOAT_TYPE,
                "SKIP_SLICING": DatasetParams.SKIP_SLICING,
                "USE_VAR_BATCH_INPUT": DatasetParams.USE_VAR_BATCH_INPUT,
                "ONE_HOT_ENCODING": DatasetParams.ONE_HOT_ENCODING,
                "USE_DOWN_SAMPLING": DatasetParams.USE_DOWN_SAMPLING,
                "USE_DELETE_PAD": LossParams.USE_DELETE_PAD,
                "USE_WEIGHTED_LOSS": LossParams.USE_WEIGHTED_LOSS,
                "LOSS_ONE_WEIGHT": LossParams.LOSS_ONE_WEIGHT,
                "LOSS_ZERO_WEIGHT": LossParams.LOSS_ZERO_WEIGHT,
                "PRED_PAD_CONST": LossParams.PRED_PAD_CONST,
                "LOSS_FUN": LossParams.getLossFunName(),
            }
            DatasetParams.COMET_EXPERIMENT.log_parameters(params)
            
            metrics = {
                "batch_size": TrainingParams.BATCH_SIZE,
                "train_duration": TrainingParams.TRAIN_DURATION,
            }
            DatasetParams.COMET_EXPERIMENT.log_metrics(dic=metrics)
            
        return
    
    @classmethod
    def trainEnsXGBModel(cls, trainDataset):
        def doTrainingEnsXGB(trainX, trainY):
            #trainX = (82663, 1, 5) || trainY= #(82663, 1, 1)
            NUM_MODELS = trainX.shape[2]
            
            model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', 
                                 learning_rate=TrainingParams.LEARNING_RATE, 
                                 n_estimators=300*NUM_MODELS, max_depth=6, n_jobs=4*NUM_MODELS, seed=50, verbosity=0,
                                 objective='binary:logistic', 
                                 reg_alpha=0, reg_lambda=1, 
                                 )
            
            trainX = np.reshape(trainX, (trainX.shape[0], NUM_MODELS))   #(82663,5)
            y_true = trainY.flatten()      #(82663,)
            model.fit(trainX, y_true)
            return model
        
        #this is for ensemble training where the inputs are predictions of individual architecture
        trainX = trainDataset[0]  #(82663, 1, 5)
        trainY = trainDataset[1]  #(82663, 1, 1)
        trainingStartTime = time.time()
        model = doTrainingEnsXGB(trainX, trainY)
        joblib.dump(model, TrainingParams.MODEL_SAVE_FILE, compress=0)
        trainingEndTime = time.time()
        TrainingParams.TRAIN_DURATION = str(datetime.timedelta(seconds=trainingEndTime-trainingStartTime))
        logger.info("Training time: {}".format(TrainingParams.TRAIN_DURATION))
        
        return
    
    @classmethod
    def trainEnsRFModel(cls, trainDataset):
        def doTrainingEnsRF(trainX, trainY):
            #trainX = (82663, 1, 5) || trainY= #(82663, 1, 1)
            NUM_MODELS = trainX.shape[2]
            
            model = RandomForestRegressor(n_estimators=300*NUM_MODELS, max_depth=2*NUM_MODELS, n_jobs=4*NUM_MODELS)
            #model = RandomForestClassifier(n_estimators=300*NUM_MODELS, max_depth=2*NUM_MODELS, n_jobs=4*NUM_MODELS,
            #                               #criterion='gini',
            #                               criterion='entropy',
            #                               class_weight='balanced'
            #                               )
            trainX = np.reshape(trainX, (trainX.shape[0], NUM_MODELS))   #(82663,5)
            y_true = trainY.flatten()      #(82663,)
            model.fit(trainX, y_true)
            return model
        
        #this is for ensemble training where the inputs are predictions of individual architecture
        trainX = trainDataset[0]  #(82663, 1, 5)
        trainY = trainDataset[1]  #(82663, 1, 1)
        trainingStartTime = time.time()
        model = doTrainingEnsRF(trainX, trainY)
        joblib.dump(model, TrainingParams.MODEL_SAVE_FILE, compress=0) 
        trainingEndTime = time.time()
        TrainingParams.TRAIN_DURATION = str(datetime.timedelta(seconds=trainingEndTime-trainingStartTime))
        logger.info("Training time: {}".format(TrainingParams.TRAIN_DURATION))
        
        return
    
    @classmethod
    def trainEnsDirModel(cls, trainDataset):
        def doTrainingEnsDir(trainX, trainY):
            #trainX = (82663, 1, 5) || trainY= #(82663, 1, 1)
            TRAIN_ITER = 10000
            NUM_MODELS = trainX.shape[2]
            
            y_true = trainY.flatten()      #(82663,)
            bestAuc = float("-inf")
            bestDirWeights = None
            for i in range(TRAIN_ITER):
                #[0.23522399 0.42992264 0.05460034 0.01380323 0.2664498 ]
                dirWeights = np.random.dirichlet(np.ones(NUM_MODELS), size=1)[0]    #(5,)
                #multiply the dirWeights[0] to the predictions of all examples of models[0], ...
                y_pred = np.array([trainX[:,:,i].flatten() * dirWeights[i] for i in range(NUM_MODELS)])    #(5, 82663)
                #sum weighted predictions of all models for each example
                y_pred = np.sum(y_pred, axis=0)   #(82663,)
                currentAuc = PPILossCls.logTrainingMetrics(i, 0.10, 0.10, y_true, y_pred)
                if currentAuc > bestAuc:
                    bestAuc = currentAuc
                    bestDirWeights = dirWeights
            return bestDirWeights
        
        #this is for ensemble training where the inputs are predictions of individual architecture
        trainX = trainDataset[0]  #(82663, 1, 5)
        trainY = trainDataset[1]  #(82663, 1, 1)
        trainingStartTime = time.time()
        ensWeights = doTrainingEnsDir(trainX, trainY)
        joblib.dump(ensWeights, TrainingParams.MODEL_SAVE_FILE, compress=0) 
        trainingEndTime = time.time()
        TrainingParams.TRAIN_DURATION = str(datetime.timedelta(seconds=trainingEndTime-trainingStartTime))
        logger.info("Training time: {}".format(TrainingParams.TRAIN_DURATION))
        
        return
    
    @classmethod
    def trainKerasModel(cls, model, trainDataset):
        #logger.info("## tf-version: " + tf.__version__ + "|| keras-version: " + tf.keras.__version__ + "|| float_type: " + K.floatx())
        logger.info("## tf-version: " + tf.__version__ + "|| float_type: " + K.floatx())
        cls.printModel(model)
        
        if trainDataset is None:
            trainX, valX, trainY, valY = PPIDatasetCls.makeDataset(DatasetParams.EXPR_TRAINING_FILE, TrainingParams.INPUT_SHAPE, TrainingParams.LABEL_SHAPE, True)
        else:   #this is for ensemble training where the inputs are predictions of individual architecture
            inputData = trainDataset[0]  #(82663, 1, 5)
            labelData = trainDataset[1]  #(82663, 1, 1)
            trainX, valX, trainY, valY = PPIDatasetCls.splitData(inputData, labelData)
        
        if not DatasetParams.USE_VAR_BATCH_INPUT:
            TrainingParams.TRAIN_SHAPE = [trainX.shape, trainY.shape]
            TrainingParams.VAL_SHAPE = [valX.shape, valY.shape]
        
        model = cls.compileModel(model)
        cls.doTraining(model, trainX, valX, trainY, valY)
        
        return
      
    @classmethod
    def trainModel(cls, model, trainDataset=None):
        trainingDateTime = datetime.datetime.now().strftime("%d-%m-%Y#%H:%M:%S")
        logger.info("\n## Starting Training & Testing;; @@ trained on: " + DatasetParams.EXPR_TRAINING_FILE + " at: " + str(trainingDateTime) + " @@")
        
        if TrainingParams.ENS_DIRICHLET_MODEL:
            cls.trainEnsDirModel(trainDataset)
        elif TrainingParams.ENS_RF_MODEL:
            cls.trainEnsRFModel(trainDataset)
        elif TrainingParams.ENS_XGBOOST_MODEL:
            cls.trainEnsXGBModel(trainDataset)
        else:
            cls.trainKerasModel(model, trainDataset)

        cls.logTrainExperiment()

        return
