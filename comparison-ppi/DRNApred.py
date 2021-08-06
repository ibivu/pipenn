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
    ALGRITHM_NAME = "DRNApred"
    DatasetParams.USE_COMET = False #True
    
    ROOT_DIR = "../data/"
    PREPARED_FASTA_N_SAMPLE_FILE = ROOT_DIR + "fasta_n_samples.txt"
    
    DRNAPRED_BIOLIP_N_FILE =  ROOT_DIR + "DRNApred_biolip_n.txt"
    DRNAPRED_ZK448_N_FILE =  ROOT_DIR + "DRNApred_zk448_n.txt"
    DRNAPRED_COLUMNS = [
                        'Amino_Acid',    
                        'probability_DNA',
                        'binary_DNA',        
                        'probability_RNA',
                        'binary_RNA',
                        ]
    DNA_PROB_COL_NAME = 'probability_DNA'
    RNA_PROB_COL_NAME = 'probability_RNA'
    
    datasetLabel = 'Biolip_N'
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

def doDRNApredModelTesting(testY, isZK):
    y_true = testY.flatten() 
    DRNApredFile = AlgParams.DRNAPRED_ZK448_N_FILE if isZK else AlgParams.DRNAPRED_BIOLIP_N_FILE
    df = pd.read_csv(DRNApredFile, delimiter = "\t")
    y_pred_dna = df.loc[:, AlgParams.DNA_PROB_COL_NAME].values
    y_pred_rna = df.loc[:, AlgParams.RNA_PROB_COL_NAME].values
    return y_true, y_pred_dna, y_pred_rna

def doTesting(testingFile, testingLabel):
    testingDateTime = datetime.datetime.now().strftime("%d-%m-%Y#%H:%M:%S")
    AlgParams.logger.info("## Testing " + testingFile + " at: " + str(testingDateTime) + " ##")
    _, testX, _, testY = PPIDatasetCls.makeDataset(testingFile, TrainingParams.INPUT_SHAPE, TrainingParams.LABEL_SHAPE, False, False)
    
    testingStartTime = time.time()
    
    isZK = True if 'ZK' in testingLabel else False  
    y_true, y_pred_dna, y_pred_rna = doDRNApredModelTesting(testY, isZK)
    predLabel = testingLabel + '_DNA' 
    y_true, y_pred_dna = PPILossCls.logTestingMetrics(y_true, y_pred_dna, predLabel)
    PPITrainTestCls.plotMetrics(testingFile, y_true, y_pred_dna, predLabel)
    predLabel = testingLabel + '_RNA' 
    y_true, y_pred_rna = PPILossCls.logTestingMetrics(y_true, y_pred_rna, predLabel)
    PPITrainTestCls.plotMetrics(testingFile, y_true, y_pred_rna, predLabel)
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

def prepareNSamplesinFasta():
    """
    Generate one FASTA file from proteins in biolip_n and zk448 to pass as an input to DRNApred.
    """
    print("## Generating nucleotide samples in FASTA for @@: " + AlgParams.PREPARED_FASTA_N_SAMPLE_FILE + " @@")
    DatasetParams.SEQ_COLUMN_NAME = 'sequence'
    df_tn = pd.read_csv(DatasetParams.PREPARED_BIOLIP_WIN_N_TESTING_FILE)
    df_zkn = pd.read_csv(DatasetParams.PREPARED_ZK448_WIN_N_BENCHMARK_FILE)
    protIds_tn = df_tn.loc[:, DatasetParams.PROT_ID_NAME].values
    protIds_zkn = df_zkn.loc[:, DatasetParams.PROT_ID_NAME].values
    seqs_tn = df_tn.loc[:, DatasetParams.SEQ_COLUMN_NAME].values
    seqs_tn = [''.join(s.replace(',', '')) for s in seqs_tn]
    seqs_zkn = df_zkn.loc[:, DatasetParams.SEQ_COLUMN_NAME].values
    seqs_zkn = [''.join(s.replace(',', '')) for s in seqs_zkn]
    num_tn_prots = len(protIds_tn)
    num_zkn_prots = len(protIds_zkn)
    with open(AlgParams.PREPARED_FASTA_N_SAMPLE_FILE, "w") as f:
        for i in range(num_tn_prots):
            f.write('>' + protIds_tn[i] + '\n')
            f.write(seqs_tn[i] + '\n')
        print(str(num_tn_prots) + ' - samples generated done for @@: ' + DatasetParams.PREPARED_BIOLIP_WIN_A_TESTING_FILE + " @@")
        for i in range(num_zkn_prots):
            f.write('>' + protIds_zkn[i] + '\n')
            f.write(seqs_zkn[i] + '\n')
        print(str(num_zkn_prots) + " - samples generated for @@: " + DatasetParams.PREPARED_ZK448_WIN_N_BENCHMARK_FILE + " @@")
    return
        
if __name__ == "__main__":
    #prepareNSamplesinFasta()
    performTesting()    