# make sure comet_ml is the first import (before all other Machine learning lib)
from comet_ml import Experiment

import datetime, time
import numpy as np
import pandas as pd

from PPILogger import PPILoggerCls
from PPILoss import PPILossCls, LossParams
from PPIDataset import PPIDatasetCls, DatasetParams
from PPITrainTest import PPITrainTestCls, TrainingParams
from PPIParams import PPIParamsCls

class AlgParams:
    ALGRITHM_NAME = "SCRIBER"
    DatasetParams.USE_COMET = False #True
    
    ROOT_DIR = "../data/"
    PREPARED_FASTA_S_SAMPLE_FILE = ROOT_DIR + "fasta_s_samples.txt"
    
    SCRIBER_ZK10_S_FILE =  ROOT_DIR + "SCRIBER_zk10_s.csv"
    SCRIBER_COLUMNS = [
                        'ResidueNumber',
                        'ResidueType',
                        'DNA-binding-propensity',
                        'RNA-binding-propensity',
                        'protein-binding-propensity',
                        'ligand-binding-propensity',
                        'fprotein-binding-propensity',
                        'protein-binding-residues'
                        ]
    SMB_PROB_COL_NAME = 'ligand-binding-propensity'
    
    datasetLabel = 'Biolip_S'
    dataset = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    LABEL_DIM = 1
    PROT_IMAGE_H, PROT_IMAGE_W = 1, 1
    INPUT_SHAPE = None  #must be determined in init.
    LABEL_SHAPE = None
    
    logger = None
    
    @classmethod
    def setShapes(cls):
        cls.INPUT_SHAPE = (cls.PROT_IMAGE_H, DatasetParams.getFeaturesDim())   
        cls.LABEL_SHAPE = (cls.PROT_IMAGE_H, cls.LABEL_DIM)
        PPIParamsCls.setShapeParams(cls.INPUT_SHAPE, cls.LABEL_SHAPE)   
        return
    
    @classmethod
    def initAlgParams(cls, dsParam=dataset, dsLabelParam=datasetLabel):
        cls.logger = PPIParamsCls.setLoggers(cls.ALGRITHM_NAME, dsLabelParam)
        PPIParamsCls.setInitParams(cls.ALGRITHM_NAME,dsParam=dsParam, dsLabelParam=dsLabelParam)
        cls.setShapes()
        return 

def doSCRIBERModelTesting(testY):
    y_true = testY.flatten() 
    SCRIBERFile = AlgParams.SCRIBER_ZK10_S_FILE
    df = pd.read_csv(SCRIBERFile)
    y_pred = df.loc[:, AlgParams.SMB_PROB_COL_NAME].values
    return y_true, y_pred

def doTesting(testingFile, testingLabel):
    testingDateTime = datetime.datetime.now().strftime("%d-%m-%Y#%H:%M:%S")
    AlgParams.logger.info("## Testing " + testingFile + " at: " + str(testingDateTime) + " ##")
    _, testX, _, testY = PPIDatasetCls.makeDataset(testingFile, TrainingParams.INPUT_SHAPE, TrainingParams.LABEL_SHAPE, False, False)
    
    testingStartTime = time.time()
    
    y_true, y_pred = doSCRIBERModelTesting(testY)
    predLabel = testingLabel + '_SMB' 
    y_true, y_pred = PPILossCls.logTestingMetrics(y_true, y_pred, predLabel)
    PPITrainTestCls.plotMetrics(testingFile, y_true, y_pred, predLabel)
    testingEndTime = time.time()
    AlgParams.logger.info("Testing time: {}".format(datetime.timedelta(seconds=testingEndTime-testingStartTime)))
    
    return 

def testModel():
    for i in range(len(DatasetParams.EXPR_TESTING_FILE_SET)):
        doTesting(DatasetParams.EXPR_TESTING_FILE_SET[i], DatasetParams.EXPR_TESTING_LABELS[i])
    PPITrainTestCls.logTestExperiment()
    return 

def performTesting():
    AlgParams.initAlgParams()
    #TrainingParams.GEN_METRICS_PER_PROT = True
    testModel()
    return 

def prepareSSamplesinFasta():
    """
    Generate one FASTA file from random 10 proteins in zk448_s to pass as an input to SCRIBER.
    """
    print("## Generating small molecule samples in FASTA for @@: " + AlgParams.PREPARED_FASTA_S_SAMPLE_FILE + " @@")
    DatasetParams.SEQ_COLUMN_NAME = 'sequence'
    df_zkn = pd.read_csv(DatasetParams.PREPARED_ZK448_WIN_S_BENCHMARK_FILE)
    df_zkn = df_zkn.sample(n=10,random_state=DatasetParams.RANDOM_STATE)
    protIds_zkn = df_zkn.loc[:, DatasetParams.PROT_ID_NAME].values
    print('prot-ids: ', protIds_zkn)
    seqs_zkn = df_zkn.loc[:, DatasetParams.SEQ_COLUMN_NAME].values
    seqs_zkn = [''.join(s.replace(',', '')) for s in seqs_zkn]
    num_zkn_prots = len(protIds_zkn)
    with open(AlgParams.PREPARED_FASTA_S_SAMPLE_FILE, "w") as f:
        for i in range(num_zkn_prots):
            f.write('>' + protIds_zkn[i] + '\n')
            f.write(seqs_zkn[i] + '\n')
    df_zkn.to_csv(DatasetParams.PREPARED_ZK10_WIN_S_BENCHMARK_FILE, index=False) 
    print(str(num_zkn_prots) + " - samples generated for @@: " + DatasetParams.PREPARED_ZK10_WIN_S_BENCHMARK_FILE + " @@")
    return
        
if __name__ == "__main__":
    #prepareSSamplesinFasta()
    performTesting()    