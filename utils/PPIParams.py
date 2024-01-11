import sys
import os

import numpy as np
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU
from tensorflow.keras.initializers import GlorotUniform, GlorotNormal, he_uniform, he_normal, lecun_uniform, lecun_normal, \
                                                 Orthogonal, TruncatedNormal, VarianceScaling

from PPILogger import PPILoggerCls
from PPIDataset import PPIDatasetCls, DatasetParams
from PPILoss import PPILossCls, LossParams
from PPITrainTest import PPITrainTestCls, TrainingParams
from PPIExplanation import PPIExplanationCls

class PPIParamsCls(object):
    EX_COLUMNS = [
                    'DYNA_q', 
                    'RSA_q', 
                    'ASA_q', 
                    'PA_q', 
                    'PB_q', 
                    'PC_q', 
                    'length',
                    #'AliSeq',
                    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 
                   ]
    
    EX_COLUMNS3 = [#old-dataset
                    #'DYNA_q', 
                    'RSA_q', 
                    'ASA_q', 
                    'PA_q', 
                    'PB_q', 
                    'PC_q', 
                    'length',
                    #'AliSeq',
                 ]
    EX_COLUMNS2 = [
                #'normalized_length',
                #'domain',
                ]
    IN_COLUMNS = [
                #'glob_stat_score',
                ]
    
    datasetLabel = 'Epitope'
    dataset = DatasetParams.FEATURE_COLUMNS_EPI_SEMI_NOWIN
    #datasetLabel = 'Biolip_N'
    #dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_NOWIN
    
    @classmethod
    # Use the following as part of "Run configuration" in Eclipse for testing web-service which requires input. 
    # Program arguments: ../../ n
    # Working directory: ${workspace_loc:nn-ppi/test/wstest}
    def setPipennParams(cls):
        pipennDatasetLabel = None
        pipennHome = os.getenv('PIPENN_HOME')
        if pipennHome is not None:
            DatasetParams.PIPENN_HOME = pipennHome
        
        nargs = len(sys.argv)
        if nargs == 3:
            # pipenn has been started with $PIPENN_HOME and $PRED_TYPE (P,N,S,A), the webservice version.
            pipennHome = sys.argv[1]
            predType = sys.argv[2]
            choices = {'p': 'UserDS_P', 'n': 'UserDS_N', 's':'UserDS_S', 'a':'UserDS_A'}
            pipennDatasetLabel = choices.get(predType)
            DatasetParams.USE_USERDS_EVAL = True
            DatasetParams.USERDS_INPUT_DIR = './'
            DatasetParams.USERDS_OUTPUT_DIR = './{}/'
            DatasetParams.PREPARED_USERDS_FILE = DatasetParams.USERDS_INPUT_DIR + DatasetParams.PREPARED_USERDS_FILE_NAME
            DatasetParams.PIPENN_HOME = pipennHome
            #print("%%%%%%%%% datasetLabel: ", pipennDatasetLabel, " | PIPENN_HOME: ", DatasetParams.PIPENN_HOME)
            print("%%%%%%%%% datasetLabel: ", pipennDatasetLabel)
        elif nargs == 2:
            # pipenn has been started with only $PIPENN_HOME, the test version for students.
            pipennHome = sys.argv[1]
            DatasetParams.USE_PIPENN_TEST = True
            DatasetParams.PIPENN_HOME = pipennHome
            DatasetParams.ROOT_DIR = './data/'
            DatasetParams.setPreparedFiles()

        print("%%%%%%%%% PIPENN_HOME: ", DatasetParams.PIPENN_HOME)    
        return pipennDatasetLabel
    
    @classmethod
    def setInitParams(cls, algorithmName, dsParam=dataset, dsExParam=EX_COLUMNS, dsInParam=IN_COLUMNS, dsLabelParam=datasetLabel):
        TrainingParams.initKeras(DatasetParams.FLOAT_TYPE)
        dsParam = dsParam + dsInParam
        dsParam = list(np.setdiff1d(np.array(dsParam), np.array(dsExParam), True))
        #DatasetParams.EXPR_DATASET_LABEL = dsLabelParam
        DatasetParams.setExprFeatures(dsParam, dsLabelParam)
        DatasetParams.setExprDatasetFiles(dsLabelParam)
        
        TrainingParams.setOutputFileNames(algorithmName)
        TrainingParams.OPT_FUN = TrainingParams.ADAM_OPT
        TrainingParams.KERNAL_INITIALIZER = he_uniform
        TrainingParams.ACTIVATION_FUN = PReLU
        TrainingParams.USE_BIAS = False
        TrainingParams.BATCH_SIZE = 8 #8 > 16 > 32 > 64
        TrainingParams.USE_EARLY_STOPPING = True        #should be true for large datasets but false for small ones.
        TrainingParams.NUM_EPOCS = 300 #200 #150
        TrainingParams.MODEL_CHECK_POINT_MODE = 1
        
        DatasetParams.USE_DOWN_SAMPLING = False
        DatasetParams.FEATURE_PAD_CONST = 0.11111111
        #DatasetParams.SKIP_SLICING = True
        #DatasetParams.USE_VAR_BATCH_INPUT = True
        
        LossParams.USE_WEIGHTED_LOSS = True
        LossParams.USE_DELETE_PAD = True #True(72.32%) #False(see below)
        LossParams.PRED_PAD_CONST = 0.0 #0.001(72.07%) #0.0(72.15%) #1.0(72.06%)
        LossParams.LOSS_ONE_WEIGHT = 0.90
        LossParams.LOSS_ZERO_WEIGHT = 0.10
        #LossParams.setLossFun(LossParams.MEAN_SQUARED)
        LossParams.setLossFun(LossParams.CROSS_ENTROPY)
        #LossParams.setLossFun(LossParams.TVERSKY)
        
        return
    
    @classmethod
    def setLoggers(cls, algorithmName, dsLabelParam):
        # !! note: setLogger MUST be called before setInitParams.
        # algorithmName is 'ensnet-ppi' if it is called from ensemble testing; otherwise it is the same as above (e.g. 'rnn-ppi')
        logger = PPILoggerCls.initLogger(algorithmName, DatasetParams.PIPENN_HOME)
        TrainingParams.ALGRITHM_NAME = algorithmName
        
        if DatasetParams.USE_COMET:
            TrainingParams.initExperiment(dsLabelParam)

        PPIDatasetCls.setLogger(logger)
        PPILossCls.setLogger(logger)
        PPITrainTestCls.setLogger(logger)
        PPIExplanationCls.setLogger(logger)
        
        return logger
    
    @classmethod
    def setShapeParams(cls, inputShape, labelShape):    
        TrainingParams.INPUT_SHAPE = inputShape
        TrainingParams.LABEL_SHAPE = labelShape
        
        return
   
