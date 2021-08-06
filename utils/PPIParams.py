import numpy as np

from tensorflow.python.keras.layers import ReLU, LeakyReLU, PReLU, ELU
from tensorflow.python.keras.initializers import GlorotUniform, GlorotNormal, he_uniform, he_normal, lecun_uniform, lecun_normal, \
                                                 Orthogonal, TruncatedNormal, VarianceScaling
from PPILogger import PPILoggerCls
from PPIDataset import PPIDatasetCls, DatasetParams
from PPILoss import PPILossCls, LossParams
from PPITrainTest import PPITrainTestCls, TrainingParams
from PPIExplanation import PPIExplanationCls

class PPIParamsCls(object):
    EX_COLUMNS = [
                #'normalized_length',
                #'domain',
                ]
    IN_COLUMNS = [
                #'glob_stat_score',
                ]
    
    #datasetLabel = 'Epitope'
    #dataset = DatasetParams.FEATURE_COLUMNS_EPI_NOWIN
    #dataset = DatasetParams.FEATURE_COLUMNS_EPI_SEMI_NOWIN
    #dataset = DatasetParams.FEATURE_COLUMNS_EPI_SEMI_WIN
    #dataset = DatasetParams.FEATURE_COLUMNS_EPI_WIN
    datasetLabel = 'Biolip_N'
    #dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_NOWIN
    dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    #datasetLabel = 'Homo_Hetro'
    #dataset = DatasetParams.FEATURE_COLUMNS_ENH_NOWIN
    #dataset = DatasetParams.FEATURE_COLUMNS_ENH_SEMI_NOWIN
    #dataset = DatasetParams.FEATURE_COLUMNS_ENH_SEMI_WIN
    #dataset = DatasetParams.FEATURE_COLUMNS_ENH_WIN
    
    @classmethod
    def setInitParams(cls, algorithmName, dsParam=dataset, dsExParam=EX_COLUMNS, dsInParam=IN_COLUMNS, dsLabelParam=datasetLabel):
        TrainingParams.initKeras(DatasetParams.FLOAT_TYPE)
        dsParam = dsParam + dsInParam
        dsParam = list(np.setdiff1d(np.array(dsParam), np.array(dsExParam), True))
        #DatasetParams.EXPR_DATASET_LABEL = dsLabelParam
        DatasetParams.setExprFeatures(dsParam, dsLabelParam)
        DatasetParams.setExprDatasetFiles(dsLabelParam)
        
        TrainingParams.setOutputFileNames(algorithmName)
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
        logger = PPILoggerCls.initLogger(algorithmName)
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
   
