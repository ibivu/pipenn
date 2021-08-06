from comet_ml import Experiment
import os
import sys
import math
import itertools as it
import functools

import pandas as pd
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras import utils as U
import tensorflow as tf

import plotly.graph_objects as go
import plotly
from scipy import stats
from sklearn.model_selection import train_test_split

# Don't use this; you get circular dependency problem.
#from PPITrainTest import TrainingParams

logger = None

class DatasetParams(object):
    USE_COMET = True
    COMET_EXPERIMENT = None
    EXPR_DATASET_LABEL = None
    EXPR_TRAINING_FILE = None
    EXPR_TESTING_FILE_SET = None
    EXPR_TESTING_LABELS = None
    EXPR_TRAINING_LABEL = None
    
    ROOT_DIR = "../data/"
    
    HOMO_TRAINING_FILE = ROOT_DIR + "homo_training.csv"
    PREPARED_HOMO_TRAINING_FILE = ROOT_DIR + "prepared_homo_training.csv"
    HOMO_TESTING_FILE = ROOT_DIR + "homo_testing.csv"
    PREPARED_HOMO_TESTING_FILE = ROOT_DIR + "prepared_homo_testing.csv"
    HETRO_TRAINING_FILE = ROOT_DIR + "hetro_training.csv"
    PREPARED_HETRO_TRAINING_FILE = ROOT_DIR + "prepared_hetro_training.csv"
    HETRO_TESTING_FILE = ROOT_DIR + "hetro_testing.csv"
    PREPARED_HETRO_TESTING_FILE = ROOT_DIR + "prepared_hetro_testing.csv"
    PREPARED_COMBINED_TRAINING_FILE = ROOT_DIR + "prepared_combined_training.csv"
    PREPARED_COMBINED_TESTING_FILE = ROOT_DIR + "prepared_combined_testing.csv"
    
    EPITOPE_TRAINING_FILE = ROOT_DIR + "epitope_training.csv"
    PREPARED_EPITOPE_TRAINING_FILE = ROOT_DIR + "prepared_epitope_training.csv"
    EPITOPE_TESTING_FILE = ROOT_DIR + "epitope_testing.csv"
    PREPARED_EPITOPE_TESTING_FILE = ROOT_DIR + "prepared_epitope_testing.csv"
    PREPARED_EPITOPE_1_TRAINING_FILE = ROOT_DIR + "epitope/expr-1/prepared_epitope_training.csv"
    PREPARED_EPITOPE_1_TESTING_FILE = ROOT_DIR + "epitope/expr-1/prepared_epitope_testing.csv"
    PREPARED_EPITOPE_2_TRAINING_FILE = ROOT_DIR + "epitope/expr-2/prepared_epitope_training.csv"
    PREPARED_EPITOPE_2_TESTING_FILE = ROOT_DIR + "epitope/expr-2/prepared_epitope_testing.csv"
    PREPARED_EPITOPE_3_TRAINING_FILE = ROOT_DIR + "epitope/expr-3/prepared_epitope_training.csv"
    PREPARED_EPITOPE_3_TESTING_FILE = ROOT_DIR + "epitope/expr-3/prepared_epitope_testing.csv"
    PREPARED_EPITOPE_4_TRAINING_FILE = ROOT_DIR + "epitope/expr-4/prepared_epitope_training.csv"
    PREPARED_EPITOPE_4_TESTING_FILE = ROOT_DIR + "epitope/expr-4/prepared_epitope_testing.csv"
    PREPARED_EPITOPE_5_TRAINING_FILE = ROOT_DIR + "epitope/expr-5/prepared_epitope_training.csv"
    PREPARED_EPITOPE_5_TESTING_FILE = ROOT_DIR + "epitope/expr-5/prepared_epitope_testing.csv"
    
    """
    # These are average-window dataset (genrated by Edis python script).
    HOMO_TRAINING_WIN_FILE = ROOT_DIR + "homo_win_training.csv"
    HOMO_TESTING_WIN_FILE = ROOT_DIR + "homo_win_testing.csv"
    HETRO_TRAINING_WIN_FILE = ROOT_DIR + "hetro_win_training.csv"
    HETRO_TESTING_WIN_FILE = ROOT_DIR + "hetro_win_testing.csv"
    """
    
    BIOLIP_WIN_TRAINING_FILE = ROOT_DIR + "biolip_win_training.csv"
    BIOLIP_WIN_TESTING_FILE = ROOT_DIR + "biolip_win_testing.csv"
    ZK448_WIN_BENCHMARK_FILE = ROOT_DIR + "ZK448_win_benchmark.csv"    
    PREPARED_BIOLIP_WIN_P_TRAINING_FILE = ROOT_DIR + "prepared_biolip_win_p_training.csv"
    PREPARED_BIOLIP_WIN_S_TRAINING_FILE = ROOT_DIR + "prepared_biolip_win_s_training.csv"
    PREPARED_BIOLIP_WIN_N_TRAINING_FILE = ROOT_DIR + "prepared_biolip_win_n_training.csv"
    PREPARED_BIOLIP_WIN_A_TRAINING_FILE = ROOT_DIR + "prepared_biolip_win_a_training.csv"
    PREPARED_BIOLIP_WIN_P_TESTING_FILE = ROOT_DIR + "prepared_biolip_win_p_testing.csv"
    PREPARED_BIOLIP_WIN_S_TESTING_FILE = ROOT_DIR + "prepared_biolip_win_s_testing.csv"
    PREPARED_BIOLIP_WIN_N_TESTING_FILE = ROOT_DIR + "prepared_biolip_win_n_testing.csv"
    PREPARED_BIOLIP_WIN_A_TESTING_FILE = ROOT_DIR + "prepared_biolip_win_a_testing.csv"
    PREPARED_ZK448_WIN_P_BENCHMARK_FILE = ROOT_DIR + "prepared_ZK448_win_p_benchmark.csv"
    PREPARED_ZK448_WIN_S_BENCHMARK_FILE = ROOT_DIR + "prepared_ZK448_win_s_benchmark.csv"
    PREPARED_ZK448_WIN_N_BENCHMARK_FILE = ROOT_DIR + "prepared_ZK448_win_n_benchmark.csv"
    PREPARED_ZK448_WIN_A_BENCHMARK_FILE = ROOT_DIR + "prepared_ZK448_win_a_benchmark.csv"
    
    #for testing models of the competitors
    PREPARED_ZK448_WIN_UA_BENCHMARK_FILE = ROOT_DIR + "prepared_ZK448_win_ua_benchmark.csv"
    PREPARED_ZK10_WIN_S_BENCHMARK_FILE = ROOT_DIR + "prepared_ZK10_win_s_benchmark.csv"
    
    PREPARED_COVID_WIN_P_BENCHMARK_FILE = ROOT_DIR + "prepared_covid_win_p_benchmark.csv"
    COVID_WIN_P_BENCHMARK_FILE = ROOT_DIR + "prepared_covid_win_p_benchmark2.csv"
    
    MISSING_VALUE_CONST = 0.11111111 #0.0 #0.00000001
    LABELS = ['I', 'NI']
    LABEL_COL_NAME = 'Interface1'
    LEN_COL_NAME = 'length'
    POS_COL_NAME = 'pos'
    REAL_LEN_COL_NAME = 'Rlength'
    PROT_ID_NAME = 'uniprot_id'
        
    PROT_VEC_ENCODINGS_FILE = ROOT_DIR + "protvec_encodings.csv"
    UNKNOWN_TRIPLET = '<unk>'
    UNKNOWN_AA = 'X' 
    
    FEATURE_COLUMNS = None
    FEATURE_COLUMNS_ENH_NOWIN = [
                    'mean_H', 
                    'H_Entropies',
                    'sd_H', 
                    'DYNA_q', 
                    'RSA_H', 
                    'RSA_q', 
                    'RSA_sd_H', 
                    'ASA_H', 
                    'ASA_q', 
                    'ASA_sd_H', 
                    'PA_H', 
                    'PA_q', 
                    'PA_sd_H', 
                    'PB_H', 
                    'PB_q', 
                    'PB_sd_H', 
                    'PC_H', 
                    'PC_q', 
                    'PC_sd_H', 
                    'length',
                    'AliSeq',
                    #'Hydropathy_Score', 
                    #'ACH_w9', 'ACH_w7', 'ACH_w5', 'ACH_w3', 
                    #'ASA_hydrophobic_total_h', 'ASA_hydrophobic_total_q', 'totalCH', 
                    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 
                   ]
    
    FEATURE_COLUMNS_ENH_SEMI_NOWIN = [
                    'DYNA_q', 
                    'RSA_q', 
                    'ASA_q', 
                    'PA_q', 
                    'PB_q', 
                    'PC_q', 
                    'length',
                    'AliSeq',
                    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 
                   ]
    
    FEATURE_COLUMNS_ENH_SEMI_WIN_cross = [
                    'domain',
                    'AliSeq',
                    'length',
                    'ASA_q',                   
                    'RSA_q',
                    'PB_q',
                    'PA_q',
                    'PC_q',
                    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',                     
                    '3_wm_ASA_q', '5_wm_ASA_q', '7_wm_ASA_q', '9_wm_ASA_q',                    
                    '3_wm_RSA_q', '5_wm_RSA_q', '7_wm_RSA_q', '9_wm_RSA_q',                    
                    '3_wm_PB_q', '5_wm_PB_q', '7_wm_PB_q', '9_wm_PB_q',
                    '3_wm_PA_q', '5_wm_PA_q', '7_wm_PA_q', '9_wm_PA_q',                                                                
                    '3_wm_PC_q', '5_wm_PC_q', '7_wm_PC_q', '9_wm_PC_q',                     
                    '3_wm_A','3_wm_R','3_wm_N','3_wm_D','3_wm_C','3_wm_Q','3_wm_E',
                    '3_wm_G','3_wm_H','3_wm_I','3_wm_L','3_wm_K','3_wm_M','3_wm_F',
                    '3_wm_P','3_wm_S','3_wm_T','3_wm_W','3_wm_Y','3_wm_V',
                    '5_wm_A','5_wm_R','5_wm_N','5_wm_D','5_wm_C','5_wm_Q','5_wm_E',
                    '5_wm_G','5_wm_H','5_wm_I','5_wm_L','5_wm_K','5_wm_M','5_wm_F',
                    '5_wm_P','5_wm_S','5_wm_T','5_wm_W','5_wm_Y','5_wm_V',
                    '7_wm_A','7_wm_R','7_wm_N','7_wm_D','7_wm_C','7_wm_Q','7_wm_E',
                    '7_wm_G','7_wm_H','7_wm_I','7_wm_L','7_wm_K','7_wm_M','7_wm_F',
                    '7_wm_P','7_wm_S','7_wm_T','7_wm_W','7_wm_Y','7_wm_V',
                    '9_wm_A','9_wm_R','9_wm_N','9_wm_D','9_wm_C','9_wm_Q','9_wm_E',
                    '9_wm_G','9_wm_H','9_wm_I','9_wm_L','9_wm_K','9_wm_M','9_wm_F',
                    '9_wm_P','9_wm_S','9_wm_T','9_wm_W','9_wm_Y','9_wm_V',
                   ]
    
    FEATURE_COLUMNS_ENH_SEMI_WIN = [
                    'DYNA_q', 
                    '3_wm_DYNA_q', '5_wm_DYNA_q', '7_wm_DYNA_q', '9_wm_DYNA_q', 
                    'RSA_q', 
                    '3_wm_RSA_q', '5_wm_RSA_q', '7_wm_RSA_q', '9_wm_RSA_q', 
                    'ASA_q', 
                    '3_wm_ASA_q', '5_wm_ASA_q', '7_wm_ASA_q', '9_wm_ASA_q', 
                    'PA_q', 
                    '3_wm_PA_q', '5_wm_PA_q', '7_wm_PA_q', '9_wm_PA_q', 
                    'PB_q', 
                    '3_wm_PB_q', '5_wm_PB_q', '7_wm_PB_q', '9_wm_PB_q', 
                    'PC_q', 
                    '3_wm_PC_q', '5_wm_PC_q', '7_wm_PC_q', '9_wm_PC_q', 
                    'length',
                    'AliSeq',
                    #'Hydropathy_Score', 
                    #'3_wm_Hydropathy_Score', '5_wm_Hydropathy_Score', '7_wm_Hydropathy_Score', '9_wm_Hydropathy_Score',
                    #'ACH_w9', 'ACH_w7', 'ACH_w5', 'ACH_w3', 
                    #'totalCH',                     #sum of Hydropathy_Score of all amino-acids of a protein
                    #'ASA_hydrophobic_total_q',     #sum of Hydrofobic_Score of all amino-acids of a protein
                    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 
                    '3_wm_A', '5_wm_A', '7_wm_A', '9_wm_A', 
                    '3_wm_R', '5_wm_R', '7_wm_R', '9_wm_R', 
                    '3_wm_N', '5_wm_N', '7_wm_N', '9_wm_N', 
                    '3_wm_D', '5_wm_D', '7_wm_D', '9_wm_D', 
                    '3_wm_C', '5_wm_C', '7_wm_C', '9_wm_C', 
                    '3_wm_Q', '5_wm_Q', '7_wm_Q', '9_wm_Q', 
                    '3_wm_E', '5_wm_E', '7_wm_E', '9_wm_E', 
                    '3_wm_G', '5_wm_G', '7_wm_G', '9_wm_G', 
                    '3_wm_H', '5_wm_H', '7_wm_H', '9_wm_H', 
                    '3_wm_I', '5_wm_I', '7_wm_I', '9_wm_I', 
                    '3_wm_L', '5_wm_L', '7_wm_L', '9_wm_L', 
                    '3_wm_K', '5_wm_K', '7_wm_K', '9_wm_K', 
                    '3_wm_M', '5_wm_M', '7_wm_M', '9_wm_M', 
                    '3_wm_F', '5_wm_F', '7_wm_F', '9_wm_F', 
                    '3_wm_P', '5_wm_P', '7_wm_P', '9_wm_P', 
                    '3_wm_S', '5_wm_S', '7_wm_S', '9_wm_S', 
                    '3_wm_T', '5_wm_T', '7_wm_T', '9_wm_T', 
                    '3_wm_W', '5_wm_W', '7_wm_W', '9_wm_W', 
                    '3_wm_Y', '5_wm_Y', '7_wm_Y', '9_wm_Y', 
                    '3_wm_V', '5_wm_V', '7_wm_V', '9_wm_V', 
                   ]
    
    FEATURE_COLUMNS_ENH_WIN = [
                    'mean_H',       #With _H hetro performs higher and homo performs lower.
                    '3_wm_mean_H', '5_wm_mean_H', '7_wm_mean_H', '9_wm_mean_H', 
                    'H_Entropies',
                    '3_wm_H_Entropies', '5_wm_H_Entropies', '7_wm_H_Entropies', '9_wm_H_Entropies', 
                    'sd_H', 
                    '3_wm_sd_H', '5_wm_sd_H', '7_wm_sd_H', '9_wm_sd_H', 
                    'DYNA_q', 
                    '3_wm_DYNA_q', '5_wm_DYNA_q', '7_wm_DYNA_q', '9_wm_DYNA_q', 
                    'RSA_H', 
                    '3_wm_RSA_H', '5_wm_RSA_H', '7_wm_RSA_H', '9_wm_RSA_H', 
                    'RSA_q', 
                    '3_wm_RSA_q', '5_wm_RSA_q', '7_wm_RSA_q', '9_wm_RSA_q', 
                    'RSA_sd_H', 
                    '3_wm_RSA_sd_H', '5_wm_RSA_sd_H', '7_wm_RSA_sd_H', '9_wm_RSA_sd_H', 
                    'ASA_H', 
                    '3_wm_ASA_H', '5_wm_ASA_H', '7_wm_ASA_H', '9_wm_ASA_H', 
                    'ASA_q', 
                    '3_wm_ASA_q', '5_wm_ASA_q', '7_wm_ASA_q', '9_wm_ASA_q', 
                    'ASA_sd_H', 
                    '3_wm_ASA_sd_H', '5_wm_ASA_sd_H', '7_wm_ASA_sd_H', '9_wm_ASA_sd_H', 
                    'PA_H', 
                    '3_wm_PA_H', '5_wm_PA_H', '7_wm_PA_H', '9_wm_PA_H', 
                    'PA_q', 
                    '3_wm_PA_q', '5_wm_PA_q', '7_wm_PA_q', '9_wm_PA_q', 
                    'PA_sd_H', 
                    '3_wm_PA_sd_H', '5_wm_PA_sd_H', '7_wm_PA_sd_H', '9_wm_PA_sd_H', 
                    'PB_H', 
                    '3_wm_PB_H', '5_wm_PB_H', '7_wm_PB_H', '9_wm_PB_H', 
                    'PB_q', 
                    '3_wm_PB_q', '5_wm_PB_q', '7_wm_PB_q', '9_wm_PB_q', 
                    'PB_sd_H', 
                    '3_wm_PB_sd_H', '5_wm_PB_sd_H', '7_wm_PB_sd_H', '9_wm_PB_sd_H', 
                    'PC_H', 
                    '3_wm_PC_H', '5_wm_PC_H', '7_wm_PC_H', '9_wm_PC_H', 
                    'PC_q', 
                    '3_wm_PC_q', '5_wm_PC_q', '7_wm_PC_q', '9_wm_PC_q', 
                    'PC_sd_H', 
                    '3_wm_PC_sd_H', '5_wm_PC_sd_H', '7_wm_PC_sd_H', '9_wm_PC_sd_H', 
                    'length',
                    'AliSeq',
                    #'Hydropathy_Score', 
                    #'3_wm_Hydropathy_Score', '5_wm_Hydropathy_Score', '7_wm_Hydropathy_Score', '9_wm_Hydropathy_Score',
                    #'ACH_w9', 'ACH_w7', 'ACH_w5', 'ACH_w3',     #average Hydropathy_Score (of neighbours) per position; differs from above
                    #'totalCH',                     #sum of Hydropathy_Score of all amino-acids of a protein
                    #'ASA_hydrophobic_total_q',     #sum of Hydrofobic_Score of all amino-acids of a protein
                    #'ASA_hydrophobic_total_h', 
                    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 
                    '3_wm_A', '5_wm_A', '7_wm_A', '9_wm_A', 
                    '3_wm_R', '5_wm_R', '7_wm_R', '9_wm_R', 
                    '3_wm_N', '5_wm_N', '7_wm_N', '9_wm_N', 
                    '3_wm_D', '5_wm_D', '7_wm_D', '9_wm_D', 
                    '3_wm_C', '5_wm_C', '7_wm_C', '9_wm_C', 
                    '3_wm_Q', '5_wm_Q', '7_wm_Q', '9_wm_Q', 
                    '3_wm_E', '5_wm_E', '7_wm_E', '9_wm_E', 
                    '3_wm_G', '5_wm_G', '7_wm_G', '9_wm_G', 
                    '3_wm_H', '5_wm_H', '7_wm_H', '9_wm_H', 
                    '3_wm_I', '5_wm_I', '7_wm_I', '9_wm_I', 
                    '3_wm_L', '5_wm_L', '7_wm_L', '9_wm_L', 
                    '3_wm_K', '5_wm_K', '7_wm_K', '9_wm_K', 
                    '3_wm_M', '5_wm_M', '7_wm_M', '9_wm_M', 
                    '3_wm_F', '5_wm_F', '7_wm_F', '9_wm_F', 
                    '3_wm_P', '5_wm_P', '7_wm_P', '9_wm_P', 
                    '3_wm_S', '5_wm_S', '7_wm_S', '9_wm_S', 
                    '3_wm_T', '5_wm_T', '7_wm_T', '9_wm_T', 
                    '3_wm_W', '5_wm_W', '7_wm_W', '9_wm_W', 
                    '3_wm_Y', '5_wm_Y', '7_wm_Y', '9_wm_Y', 
                    '3_wm_V', '5_wm_V', '7_wm_V', '9_wm_V', 
                    ]
    
    FEATURE_COLUMNS_EPI_NOWIN = [
                   'mean_H',
                   'H_Entropies',
                   'sd_H',
                   'DYNA_q',
                   'RSA_H', 
                   'RSA_q', 
                   'RSA_sd_H',
                   'ASA_H', 
                   'ASA_q', 
                   'ASA_sd_H',
                   'PA_H',  
                   'PA_q', 
                   'PA_sd_H',
                   'PB_H', 
                   'PB_q', 
                   'PB_sd_H',
                   'PC_H',  
                   'PC_q', 
                   'PC_sd_H',
                   'length', 
                   'AliSeq',
                   ]
    
        
    FEATURE_COLUMNS_EPI_SEMI_NOWIN = [
                    'DYNA_q', 
                    'RSA_q', 
                    'ASA_q', 
                    'PA_q', 
                    'PB_q', 
                    'PC_q', 
                    'length',
                    'AliSeq',
                   ]
    
    FEATURE_COLUMNS_EPI_SEMI_WIN = [
                    'DYNA_q', 
                    '3_wm_DYNA_q', '5_wm_DYNA_q', '7_wm_DYNA_q', '9_wm_DYNA_q', 
                    'RSA_q', 
                    '3_wm_RSA_q', '5_wm_RSA_q', '7_wm_RSA_q', '9_wm_RSA_q', 
                    'ASA_q', 
                    '3_wm_ASA_q', '5_wm_ASA_q', '7_wm_ASA_q', '9_wm_ASA_q', 
                    'PA_q', 
                    '3_wm_PA_q', '5_wm_PA_q', '7_wm_PA_q', '9_wm_PA_q', 
                    'PB_q', 
                    '3_wm_PB_q', '5_wm_PB_q', '7_wm_PB_q', '9_wm_PB_q', 
                    'PC_q', 
                    '3_wm_PC_q', '5_wm_PC_q', '7_wm_PC_q', '9_wm_PC_q', 
                    'length',
                    'AliSeq',
                   ]
    
    FEATURE_COLUMNS_EPI_WIN = [
                   'mean_H',       #With _H hetro performs higher and homo performs lower.
                    '3_wm_mean_H', '5_wm_mean_H', '7_wm_mean_H', '9_wm_mean_H', 
                    'H_Entropies',
                    '3_wm_H_Entropies', '5_wm_H_Entropies', '7_wm_H_Entropies', '9_wm_H_Entropies', 
                    'sd_H', 
                    '3_wm_sd_H', '5_wm_sd_H', '7_wm_sd_H', '9_wm_sd_H', 
                    'DYNA_q', 
                    '3_wm_DYNA_q', '5_wm_DYNA_q', '7_wm_DYNA_q', '9_wm_DYNA_q', 
                    'RSA_H', 
                    '3_wm_RSA_H', '5_wm_RSA_H', '7_wm_RSA_H', '9_wm_RSA_H', 
                    'RSA_q', 
                    '3_wm_RSA_q', '5_wm_RSA_q', '7_wm_RSA_q', '9_wm_RSA_q', 
                    'RSA_sd_H', 
                    '3_wm_RSA_sd_H', '5_wm_RSA_sd_H', '7_wm_RSA_sd_H', '9_wm_RSA_sd_H', 
                    'ASA_H', 
                    '3_wm_ASA_H', '5_wm_ASA_H', '7_wm_ASA_H', '9_wm_ASA_H', 
                    'ASA_q', 
                    '3_wm_ASA_q', '5_wm_ASA_q', '7_wm_ASA_q', '9_wm_ASA_q', 
                    'ASA_sd_H', 
                    '3_wm_ASA_sd_H', '5_wm_ASA_sd_H', '7_wm_ASA_sd_H', '9_wm_ASA_sd_H', 
                    'PA_H', 
                    '3_wm_PA_H', '5_wm_PA_H', '7_wm_PA_H', '9_wm_PA_H', 
                    'PA_q', 
                    '3_wm_PA_q', '5_wm_PA_q', '7_wm_PA_q', '9_wm_PA_q', 
                    'PA_sd_H', 
                    '3_wm_PA_sd_H', '5_wm_PA_sd_H', '7_wm_PA_sd_H', '9_wm_PA_sd_H', 
                    'PB_H', 
                    '3_wm_PB_H', '5_wm_PB_H', '7_wm_PB_H', '9_wm_PB_H', 
                    'PB_q', 
                    '3_wm_PB_q', '5_wm_PB_q', '7_wm_PB_q', '9_wm_PB_q', 
                    'PB_sd_H', 
                    '3_wm_PB_sd_H', '5_wm_PB_sd_H', '7_wm_PB_sd_H', '9_wm_PB_sd_H', 
                    'PC_H', 
                    '3_wm_PC_H', '5_wm_PC_H', '7_wm_PC_H', '9_wm_PC_H', 
                    'PC_q', 
                    '3_wm_PC_q', '5_wm_PC_q', '7_wm_PC_q', '9_wm_PC_q', 
                    'PC_sd_H', 
                    '3_wm_PC_sd_H', '5_wm_PC_sd_H', '7_wm_PC_sd_H', '9_wm_PC_sd_H', 
                    'length',
                    'AliSeq',
                   ]
        
    FEATURE_COLUMNS_STD_NOWIN = [
                   'mean_H',
                   'H_Entropies',
                   'sd_H',
                   'DYNA_q',
                   'RSA_H', 
                   'RSA_q', 
                   'RSA_sd_H',
                   'ASA_H', 
                   'ASA_q', 
                   'ASA_sd_H',
                   'PA_H',  
                   'PA_q', 
                   'PA_sd_H',
                   'PB_H', 
                   'PB_q', 
                   'PB_sd_H',
                   'PC_H',  
                   'PC_q', 
                   'PC_sd_H',
                   'length', 
                   'AliSeq',
                   ]
        
    FEATURE_COLUMNS_STD_SEMI_NOWIN = [
                   'DYNA_q',
                   'RSA_q', 
                   'ASA_q', 
                   'PA_q', 
                   'PB_q', 
                   'PC_q', 
                   'length',
                   'AliSeq',
                   ]
    
    FEATURE_COLUMNS_STD_SEMI_WIN = [
                   'DYNA_q',
                   'X1DYNA_q', 'X2DYNA_q', 'X3DYNA_q', 'X4DYNA_q', 'X5DYNA_q', 'X6DYNA_q', 'X7DYNA_q', 'X8DYNA_q', 
                   'RSA_q', 
                   'X1RSA_q', 'X2RSA_q', 'X3RSA_q', 'X4RSA_q', 'X5RSA_q', 'X6RSA_q', 'X7RSA_q', 'X8RSA_q', 
                   'ASA_q', 
                   'X1ASA_q', 'X2ASA_q', 'X3ASA_q', 'X4ASA_q', 'X5ASA_q', 'X6ASA_q', 'X7ASA_q', 'X8ASA_q', 
                   'PA_q', 
                   'X1PA_q', 'X2PA_q', 'X3PA_q', 'X4PA_q', 'X5PA_q', 'X6PA_q', 'X7PA_q', 'X8PA_q', 
                   'PB_q', 
                   'X1PB_q', 'X2PB_q', 'X3PB_q', 'X4PB_q', 'X5PB_q', 'X6PB_q', 'X7PB_q', 'X8PB_q', 
                   'PC_q', 
                   'X1PC_q', 'X2PC_q', 'X3PC_q', 'X4PC_q', 'X5PC_q', 'X6PC_q', 'X7PC_q', 'X8PC_q',
                   'length',
                   'AliSeq',
                   ]
    
    FEATURE_COLUMNS_STD_WIN = [
                   'mean_H',
                   'X1mean_H', 'X2mean_H', 'X3mean_H', 'X4mean_H', 'X5mean_H', 'X6mean_H', 'X7mean_H', 'X8mean_H', 
                   'H_Entropies',
                   'X1H_Entropies', 'X2H_Entropies', 'X3H_Entropies', 'X4H_Entropies', 'X5H_Entropies', 'X6H_Entropies', 'X7H_Entropies', 'X8H_Entropies', 
                   'sd_H',
                   'X1sd_H', 'X2sd_H', 'X3sd_H', 'X4sd_H', 'X5sd_H', 'X6sd_H', 'X7sd_H', 'X8sd_H', 
                   'DYNA_q',
                   'X1DYNA_q', 'X2DYNA_q', 'X3DYNA_q', 'X4DYNA_q', 'X5DYNA_q', 'X6DYNA_q', 'X7DYNA_q', 'X8DYNA_q', 
                   'RSA_H', 
                   'X1RSA_H', 'X2RSA_H', 'X3RSA_H', 'X4RSA_H', 'X5RSA_H', 'X6RSA_H', 'X7RSA_H', 'X8RSA_H', 
                   'RSA_q', 
                   'X1RSA_q', 'X2RSA_q', 'X3RSA_q', 'X4RSA_q', 'X5RSA_q', 'X6RSA_q', 'X7RSA_q', 'X8RSA_q', 
                   'RSA_sd_H',
                   'X1RSA_sd_H', 'X2RSA_sd_H', 'X3RSA_sd_H', 'X4RSA_sd_H', 'X5RSA_sd_H', 'X6RSA_sd_H', 'X7RSA_sd_H', 'X8RSA_sd_H', 
                   'ASA_H', 
                   'X1ASA_H', 'X2ASA_H', 'X3ASA_H', 'X4ASA_H', 'X5ASA_H', 'X6ASA_H', 'X7ASA_H', 'X8ASA_H', 
                   'ASA_q', 
                   'X1ASA_q', 'X2ASA_q', 'X3ASA_q', 'X4ASA_q', 'X5ASA_q', 'X6ASA_q', 'X7ASA_q', 'X8ASA_q', 
                   'ASA_sd_H',
                   'X1ASA_sd_H', 'X2ASA_sd_H', 'X3ASA_sd_H', 'X4ASA_sd_H', 'X5ASA_sd_H', 'X6ASA_sd_H', 'X7ASA_sd_H', 'X8ASA_sd_H', 
                   'PA_H',  
                   'X1PA_H', 'X2PA_H', 'X3PA_H', 'X4PA_H', 'X5PA_H', 'X6PA_H', 'X7PA_H', 'X8PA_H', 
                   'PA_q', 
                   'X1PA_q', 'X2PA_q', 'X3PA_q', 'X4PA_q', 'X5PA_q', 'X6PA_q', 'X7PA_q', 'X8PA_q', 
                   'PA_sd_H',
                   'X1PA_sd_H', 'X2PA_sd_H', 'X3PA_sd_H', 'X4PA_sd_H', 'X5PA_sd_H', 'X6PA_sd_H', 'X7PA_sd_H', 'X8PA_sd_H', 
                   'PB_H', 
                   'X1PB_H', 'X2PB_H', 'X3PB_H', 'X4PB_H', 'X5PB_H', 'X6PB_H', 'X7PB_H', 'X8PB_H', 
                   'PB_q', 
                   'X1PB_q', 'X2PB_q', 'X3PB_q', 'X4PB_q', 'X5PB_q', 'X6PB_q', 'X7PB_q', 'X8PB_q', 
                   'PB_sd_H',
                   'X1PB_sd_H', 'X2PB_sd_H', 'X3PB_sd_H', 'X4PB_sd_H', 'X5PB_sd_H', 'X6PB_sd_H', 'X7PB_sd_H', 'X8PB_sd_H',
                   'PC_H',  
                   'X1PC_H', 'X2PC_H', 'X3PC_H', 'X4PC_H', 'X5PC_H', 'X6PC_H', 'X7PC_H', 'X8PC_H', 
                   'PC_q', 
                   'X1PC_q', 'X2PC_q', 'X3PC_q', 'X4PC_q', 'X5PC_q', 'X6PC_q', 'X7PC_q', 'X8PC_q',
                   'PC_sd_H', 
                   'X1PC_sd_H', 'X2PC_sd_H', 'X3PC_sd_H', 'X4PC_sd_H', 'X5PC_sd_H', 'X6PC_sd_H', 'X7PC_sd_H', 'X8PC_sd_H', 
                   'length',
                   'AliSeq',
                   ]
   
    FEATURE_COLUMNS_BIOLIP_NOWIN = [
                    #'p_interface',
                    #'s_interface',
                    #'n_interface',
                    #'any_interface',
                    #'length',
                    #'uniprot_date',
                    #'uniprot_id',
                    #'domain_name',
                    #'domain_id',
                    #'abs_surf_acc','3_wm_abs_surf_acc','5_wm_abs_surf_acc','7_wm_abs_surf_acc','9_wm_abs_surf_acc',
                    #'hydropathy_index','3_wm_hydropathy_index','5_wm_hydropathy_index','7_wm_hydropathy_index','9_wm_hydropathy_index',
                    #'ACH_w3','ACH_w5','ACH_w7','ACH_w9',
                    
                    'domain',
                    'sequence',
                    'normalized_length',
                    'normalized_abs_surf_acc',
                    #'normalized_hydropathy_index',
                    'rel_surf_acc',
                    'prob_sheet',
                    'prob_helix',
                    'prob_coil',
                    'pssm_A','pssm_R','pssm_N','pssm_D','pssm_C','pssm_Q','pssm_E','pssm_G','pssm_H','pssm_I',
                    'pssm_L','pssm_K','pssm_M','pssm_F','pssm_P','pssm_S','pssm_T','pssm_W','pssm_Y','pssm_V',
                    ]
    
    FEATURE_COLUMNS_BIOLIP_WIN_cross = [
                    'DYNA_q', 
                    '3_wm_DYNA_q', '5_wm_DYNA_q', '7_wm_DYNA_q', '9_wm_DYNA_q',
                    'rel_surf_acc',
                    '3_wm_rel_surf_acc','5_wm_rel_surf_acc','7_wm_rel_surf_acc','9_wm_rel_surf_acc',
                    'normalized_abs_surf_acc',
                    '3_wm_normalized_abs_surf_acc','5_wm_normalized_abs_surf_acc','7_wm_normalized_abs_surf_acc','9_wm_normalized_abs_surf_acc',
                    'prob_helix',
                    '3_wm_prob_helix','5_wm_prob_helix','7_wm_prob_helix','9_wm_prob_helix',
                    'prob_sheet', 
                    '3_wm_prob_sheet','5_wm_prob_sheet','7_wm_prob_sheet','9_wm_prob_sheet',
                    'prob_coil',
                    '3_wm_prob_coil','5_wm_prob_coil','7_wm_prob_coil','9_wm_prob_coil',
                    'normalized_length',
                    #'length', #if you want to normalize length between 20-2050.                    
                    'sequence',
                    'pssm_A','pssm_R','pssm_N','pssm_D','pssm_C','pssm_Q','pssm_E','pssm_G','pssm_H','pssm_I',
                    'pssm_L','pssm_K','pssm_M','pssm_F','pssm_P','pssm_S','pssm_T','pssm_W','pssm_Y','pssm_V',
                    '3_wm_pssm_A', '5_wm_pssm_A', '7_wm_pssm_A', '9_wm_pssm_A', 
                    '3_wm_pssm_R', '5_wm_pssm_R', '7_wm_pssm_R', '9_wm_pssm_R', 
                    '3_wm_pssm_N', '5_wm_pssm_N', '7_wm_pssm_N', '9_wm_pssm_N', 
                    '3_wm_pssm_D', '5_wm_pssm_D', '7_wm_pssm_D', '9_wm_pssm_D', 
                    '3_wm_pssm_C', '5_wm_pssm_C', '7_wm_pssm_C', '9_wm_pssm_C', 
                    '3_wm_pssm_Q', '5_wm_pssm_Q', '7_wm_pssm_Q', '9_wm_pssm_Q', 
                    '3_wm_pssm_E', '5_wm_pssm_E', '7_wm_pssm_E', '9_wm_pssm_E', 
                    '3_wm_pssm_G', '5_wm_pssm_G', '7_wm_pssm_G', '9_wm_pssm_G', 
                    '3_wm_pssm_H', '5_wm_pssm_H', '7_wm_pssm_H', '9_wm_pssm_H', 
                    '3_wm_pssm_I', '5_wm_pssm_I', '7_wm_pssm_I', '9_wm_pssm_I', 
                    '3_wm_pssm_L', '5_wm_pssm_L', '7_wm_pssm_L', '9_wm_pssm_L', 
                    '3_wm_pssm_K', '5_wm_pssm_K', '7_wm_pssm_K', '9_wm_pssm_K', 
                    '3_wm_pssm_M', '5_wm_pssm_M', '7_wm_pssm_M', '9_wm_pssm_M', 
                    '3_wm_pssm_F', '5_wm_pssm_F', '7_wm_pssm_F', '9_wm_pssm_F', 
                    '3_wm_pssm_P', '5_wm_pssm_P', '7_wm_pssm_P', '9_wm_pssm_P', 
                    '3_wm_pssm_S', '5_wm_pssm_S', '7_wm_pssm_S', '9_wm_pssm_S', 
                    '3_wm_pssm_T', '5_wm_pssm_T', '7_wm_pssm_T', '9_wm_pssm_T', 
                    '3_wm_pssm_W', '5_wm_pssm_W', '7_wm_pssm_W', '9_wm_pssm_W', 
                    '3_wm_pssm_Y', '5_wm_pssm_Y', '7_wm_pssm_Y', '9_wm_pssm_Y', 
                    '3_wm_pssm_V', '5_wm_pssm_V', '7_wm_pssm_V', '9_wm_pssm_V', 
                    ]
    FEATURE_COLUMNS_BIOLIP_WIN = [
                    #always put the 'sequence' (aa), and all one-hot encoded features, at the end because of SHAP aggregation.
                    
                    #'p_interface',
                    #'s_interface',
                    #'n_interface',
                    #'any_interface',
                    #'uniprot_date',
                    #'uniprot_id',
                    #'domain_name',
                    #'domain_id',
                    #'abs_surf_acc','3_wm_abs_surf_acc','5_wm_abs_surf_acc','7_wm_abs_surf_acc','9_wm_abs_surf_acc',
                    #'hydropathy_index','3_wm_hydropathy_index','5_wm_hydropathy_index','7_wm_hydropathy_index','9_wm_hydropathy_index',
                    #'ACH_w3','ACH_w5','ACH_w7','ACH_w9',
                    
                    'domain',
                    'sequence',
                    'normalized_length',
                    #'length',
                    'normalized_abs_surf_acc',
                    #'normalized_hydropathy_index',     #This makes the performance lower.
                    #'3_wm_normalized_hydropathy_index','5_wm_normalized_hydropathy_index','7_wm_normalized_hydropathy_index','9_wm_normalized_hydropathy_index',
                    'rel_surf_acc',
                    'prob_sheet',
                    'prob_helix',
                    'prob_coil',
                    'pssm_A','pssm_R','pssm_N','pssm_D','pssm_C','pssm_Q','pssm_E','pssm_G','pssm_H','pssm_I',
                    'pssm_L','pssm_K','pssm_M','pssm_F','pssm_P','pssm_S','pssm_T','pssm_W','pssm_Y','pssm_V',
                    '3_wm_normalized_abs_surf_acc','5_wm_normalized_abs_surf_acc','7_wm_normalized_abs_surf_acc','9_wm_normalized_abs_surf_acc',
                    '3_wm_rel_surf_acc','5_wm_rel_surf_acc','7_wm_rel_surf_acc','9_wm_rel_surf_acc',
                    '3_wm_prob_sheet','5_wm_prob_sheet','7_wm_prob_sheet','9_wm_prob_sheet',
                    '3_wm_prob_helix','5_wm_prob_helix','7_wm_prob_helix','9_wm_prob_helix',
                    '3_wm_prob_coil','5_wm_prob_coil','7_wm_prob_coil','9_wm_prob_coil',
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
                    ]
    GLOBAL_FEATURE_COLUMNS = [
                    'normalized_length',
                    ]
    
    LABEL_COLUMNS = ['Interface1']
    NORMALIZABLE_COLUMNS = ['length', 
                            'ASA_H', 
                            'X1ASA_H', 'X2ASA_H', 'X3ASA_H', 'X4ASA_H', 'X5ASA_H', 'X6ASA_H', 'X7ASA_H', 'X8ASA_H', 
                            '3_wm_ASA_H', '5_wm_ASA_H', '7_wm_ASA_H', '9_wm_ASA_H', 
                            'ASA_q', 
                            'X1ASA_q', 'X2ASA_q', 'X3ASA_q', 'X4ASA_q', 'X5ASA_q', 'X6ASA_q', 'X7ASA_q', 'X8ASA_q', 
                            '3_wm_ASA_q', '5_wm_ASA_q', '7_wm_ASA_q', '9_wm_ASA_q', 
                            'ASA_sd_H', 
                            'X1ASA_sd_H', 'X2ASA_sd_H', 'X3ASA_sd_H', 'X4ASA_sd_H', 'X5ASA_sd_H', 'X6ASA_sd_H', 'X7ASA_sd_H', 'X8ASA_sd_H',
                            '3_wm_ASA_sd_H', '5_wm_ASA_sd_H', '7_wm_ASA_sd_H', '3_wm_ASA_sd_H', 
                            'Hydropathy_Score',
                            '3_wm_Hydropathy_Score', '5_wm_Hydropathy_Score', '7_wm_Hydropathy_Score', '9_wm_Hydropathy_Score',
                            'ACH_w9', 'ACH_w7', 'ACH_w5', 'ACH_w3', 
                            'ASA_hydrophobic_total_h', 'ASA_hydrophobic_total_q',
                            'totalCH', 'totalCH_hydrophobic',
                            ]
    
    ALL_COLUMNS = None
    FEATURES_SIZE = None
    SEQ_COLUMN_NAME = 'AliSeq'
    SEQ_COLUMN_IND = None
    ONE_HOT_ENCODING = True
    AA_LIST =  ['A', 'C', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'N', 'Q', 'P', 'S', 'R', 'T', 'W', 'V', 'Y', 'X']
    PROT_RESIDU_ENCODING_DIM = None
    FEATURES_SIZE_COMPENSATION = None
    PROT_IMAGE_C = None
    USE_VAR_BATCH_INPUT = False
    MAX_PROT_LEN_PER_BATCH = None
    USE_DOWN_SAMPLING = False
    DOWN_SAMPLING_THRESHOLD = 0.3
    DOWN_SAMPLING_STD = 2
    SKIP_SLICING = False
    USE_DATA_AUGMENTATION = False
    SHIFT_WEIGHT = 1
    FEATURE_PAD_CONST = 0
    LABEL_PAD_CONST = 5
    REAL_TRAIN_PROT_LENS = None
    REAL_VAL_PROT_LENS = None
    PROT_IDS = None     #list of prot_ids used to save predictions (for ensemble methods)
    VALIDATION_RATE = 0.2 
    RANDOM_STATE = 50
    # Default float-type of keras/tensorflow is float32. In this way, we can control the float-type to avoid type-conflict problems.
    # Besides, float64 performs better than float32.
    FLOAT_TYPE = 'float64' #'float32'
    
    USE_YOLO = False
    GRID_CELL_H, GRID_CELL_W = None, None
    GRID_SIZE = None
    # Number of grids
    GRID_H, GRID_W = None, None
    # Number of patterns of a binary matrix of length-n (p) including zero-matrix = 2^(n*n); if n=2 then p=2^(2*2)=16; if n=3 then p=2^(3*2)=512
    NUM_PATTERNS = None
    # Number of classes; all patterns except for zero. 
    NUM_CLASSES = None
    # Number of boxes a cell can do predictions.
    NUM_BOXES = 1
    # A bounding box will be represented as [bx,by,bh,bw,pc]
    BOX_DIM = 5
    # We define one pattern (box) per grid-cell. So, the bounding box will be: [bx,by,bh,bw,pc,c1,...,]
    YOLO_LABEL_DIM = None
    YOLO_LABEL_SHAPE = None
    YOLO_OUTPUT_KERNAL_SIZE = None
    YOLO_NORM_H = None
    YOLO_NORM_W = None
    YOLO_PATTERNS = None
    YOLO_BH = None
    YOLO_BW = None
    
    USE_ATT = False
    ATT_INPUT_LEN = None
    ATT_PAT_LEN = None
    ATT_PATTERNS = None
    ATT_LABEL_DIM = None
    ATT_LABEL_LEN = None
    ATT_EXT_LABEL_LEN = None
    ATT_LABEL_SHAPE = None
    ATT_EXT_LABEL_SHAPE = None
    ATT_DECODER_START = None
    ATT_DECODER_END = None
    ATT_DECODER_INPUT_LEN = None
    ATT_DECODER_INPUT_SHAPE = None
    ATT_DECODER_INF_INPUT_SHAPE = None
    
    USE_MC_RESNET = False
    MC_RESNET_INPUT_LEN = None
    MC_RESNET_PAT_LEN = None
    MC_RESNET_PATTERNS = None
    MC_RESNET_LABEL_DIM = None
    MC_RESNET_LABEL_LEN = None
    
    @classmethod
    def setExprDatasetFiles(cls, dsLabel):
        trainingFileDict = {
            'Cross_BL_A_HH': DatasetParams.PREPARED_BIOLIP_WIN_A_TRAINING_FILE,
            'Cross_BL_P_HH': DatasetParams.PREPARED_BIOLIP_WIN_P_TRAINING_FILE,
            'Cross_HH_BL_P': DatasetParams.PREPARED_COMBINED_TRAINING_FILE,
            'HH_Combined': DatasetParams.PREPARED_COMBINED_TRAINING_FILE,
            'Homo_Hetro': DatasetParams.PREPARED_COMBINED_TRAINING_FILE,
            'Homo': DatasetParams.PREPARED_HOMO_TRAINING_FILE,
            'Hetro': DatasetParams.PREPARED_HETRO_TRAINING_FILE,
            'Biolip_P': DatasetParams.PREPARED_BIOLIP_WIN_P_TRAINING_FILE,
            'Biolip_S': DatasetParams.PREPARED_BIOLIP_WIN_S_TRAINING_FILE,
            'Biolip_N': DatasetParams.PREPARED_BIOLIP_WIN_N_TRAINING_FILE,
            'Biolip_A': DatasetParams.PREPARED_BIOLIP_WIN_A_TRAINING_FILE,
            'Epitope': DatasetParams.PREPARED_EPITOPE_TRAINING_FILE,
        }
        testingFileDict = {
            'Cross_BL_A_HH': [DatasetParams.PREPARED_HOMO_TESTING_FILE, DatasetParams.PREPARED_HETRO_TESTING_FILE],
            'Cross_BL_P_HH': [DatasetParams.PREPARED_HOMO_TESTING_FILE, DatasetParams.PREPARED_HETRO_TESTING_FILE],
            'Cross_HH_BL_P': [DatasetParams.PREPARED_ZK448_WIN_P_BENCHMARK_FILE, DatasetParams.PREPARED_BIOLIP_WIN_P_TESTING_FILE],
            'HH_Combined': [DatasetParams.PREPARED_COMBINED_TESTING_FILE],
            'Homo_Hetro': [DatasetParams.PREPARED_HOMO_TESTING_FILE, DatasetParams.PREPARED_HETRO_TESTING_FILE],
            'Homo': [DatasetParams.PREPARED_HOMO_TESTING_FILE, DatasetParams.PREPARED_HETRO_TESTING_FILE],
            'Hetro': [DatasetParams.PREPARED_HOMO_TESTING_FILE, DatasetParams.PREPARED_HETRO_TESTING_FILE],
            'Biolip_P': [DatasetParams.PREPARED_ZK448_WIN_P_BENCHMARK_FILE, DatasetParams.PREPARED_BIOLIP_WIN_P_TESTING_FILE],
            'Biolip_S': [DatasetParams.PREPARED_ZK448_WIN_S_BENCHMARK_FILE, DatasetParams.PREPARED_BIOLIP_WIN_S_TESTING_FILE],
            #'Biolip_S': [DatasetParams.PREPARED_ZK10_WIN_S_BENCHMARK_FILE],
            'Biolip_N': [DatasetParams.PREPARED_ZK448_WIN_N_BENCHMARK_FILE, DatasetParams.PREPARED_BIOLIP_WIN_N_TESTING_FILE],
            'Biolip_A': [DatasetParams.PREPARED_ZK448_WIN_A_BENCHMARK_FILE, DatasetParams.PREPARED_BIOLIP_WIN_A_TESTING_FILE],
            #'Biolip_A': [DatasetParams.PREPARED_ZK448_WIN_UA_BENCHMARK_FILE],
            'Epitope': [DatasetParams.PREPARED_EPITOPE_TESTING_FILE],
            'COVID_P': [DatasetParams.PREPARED_COVID_WIN_P_BENCHMARK_FILE],
        }
        testingLabelDict = {
            'Cross_BL_A_HH': ['Homo', 'Hetro'],
            'Cross_BL_P_HH': ['Homo', 'Hetro'],
            'Cross_HH_BL_P':  ['ZK448_P', 'BLTEST_P'],
            'HH_Combined': ['HH_Combined'],
            'Homo_Hetro': ['Homo', 'Hetro'],
            'Homo': ['Homo', 'Hetro'],
            'Hetro': ['Homo', 'Hetro'],
            'Biolip_P': ['ZK448_P', 'BLTEST_P'],
            'Biolip_S': ['ZK448_S', 'BLTEST_S'],
            #'Biolip_S': ['ZK10_S'],
            'Biolip_N': ['ZK448_N', 'BLTEST_N'],
            'Biolip_A': ['ZK448_A', 'BLTEST_A'],
            #'Biolip_A': ['ZK448_UA'],
            'Epitope': ['EPI_TEST'],
            'COVID_P': ['COVID_P'],
        }
        trainingLabelDict = {
            'Cross_BL_A_HH': 'Cross_BL_A_HH',
            'Cross_BL_P_HH': 'Cross_BL_P_HH',
            'Cross_HH_BL_P': 'Cross_HH_BL_P',
            'HH_Combined': 'HH_Combined',
            'Homo_Hetro': 'Homo_Hetro',
            'Homo': 'Homo',
            'Hetro': 'Hetro',
            'Biolip_P': 'Biolip_P',
            'Biolip_S': 'Biolip_S',
            'Biolip_N': 'Biolip_N',
            'Biolip_A': 'Biolip_A',
            'Epitope': 'Epitope',
        }
        cls.EXPR_DATASET_LABEL = dsLabel
        cls.EXPR_TRAINING_FILE = trainingFileDict.get(cls.EXPR_DATASET_LABEL)
        cls.EXPR_TESTING_FILE_SET = testingFileDict.get(cls.EXPR_DATASET_LABEL)
        cls.EXPR_TESTING_LABELS = testingLabelDict.get(cls.EXPR_DATASET_LABEL)
        cls.EXPR_TRAINING_LABEL = trainingLabelDict.get(cls.EXPR_DATASET_LABEL)
        
        return
    
    @classmethod
    def setCondParams(cls):
        cls.FEATURES_SIZE = len(cls.FEATURE_COLUMNS)
        try:
            cls.SEQ_COLUMN_IND = cls.FEATURE_COLUMNS.index(cls.SEQ_COLUMN_NAME)
            # One-Hot dimension = 21; Prot2vec dimension = 100
            if cls.ONE_HOT_ENCODING:
                cls.PROT_RESIDU_ENCODING_DIM = len(cls.AA_LIST)
            else:
                cls.PROT_RESIDU_ENCODING_DIM = 100    
            # Note that encoding replaces AliSeq; so we have to remove AliSeq from the FEATURES_SIZE.
            cls.FEATURES_SIZE_COMPENSATION = -1
        except ValueError:
            cls.SEQ_COLUMN_IND = 1000 
            cls.PROT_RESIDU_ENCODING_DIM = 0 
            cls.FEATURES_SIZE_COMPENSATION = 0
        
        cls.PROT_IMAGE_C = cls.FEATURES_SIZE + cls.PROT_RESIDU_ENCODING_DIM + cls.FEATURES_SIZE_COMPENSATION
    
    @classmethod
    def setAttParams(cls, inputLen, attPatLen):
            # Note that the inputLen must be divisible by patLen.
            cls.ATT_INPUT_LEN = inputLen                                                    #576
            cls.ATT_PAT_LEN = attPatLen                                                     #6
            # We extend patterns with one bit to represent START and END indicators.
            cls.NUM_CLASSES = 2 ** (cls.ATT_PAT_LEN + 1)                                    #128
            cls.ATT_LABEL_DIM = cls.NUM_CLASSES                                             #128
            cls.ATT_LABEL_LEN = cls.ATT_INPUT_LEN // cls.ATT_PAT_LEN                        #96
            cls.ATT_LABEL_SHAPE = (cls.ATT_LABEL_LEN, cls.ATT_LABEL_DIM)                    #(96,128)
            cls.ATT_EXT_LABEL_LEN = cls.ATT_LABEL_LEN + 2                                   #98
            cls.ATT_EXT_LABEL_SHAPE = (cls.ATT_EXT_LABEL_LEN, cls.ATT_LABEL_DIM)            #(98,128)
            cls.ATT_DECODER_INPUT_LEN = cls.ATT_LABEL_LEN + 1                               #97
            cls.ATT_DECODER_INPUT_SHAPE = (cls.ATT_DECODER_INPUT_LEN, cls.ATT_LABEL_DIM)    #(97,128)
            cls.ATT_DECODER_INF_INPUT_SHAPE = (1, cls.ATT_LABEL_DIM)                        #(1,128)
    
    @classmethod
    def setMCResnetParams(cls, inputLen, patLen):
            # Note that the inputLen must be divisible by patLen.
            cls.MC_RESNET_INPUT_LEN = inputLen                                                    #576
            cls.MC_RESNET_PAT_LEN = patLen                                                        #6
            cls.NUM_CLASSES = 2 ** cls.MC_RESNET_PAT_LEN                                          #128
            cls.MC_RESNET_LABEL_DIM = cls.NUM_CLASSES                                             #128
            cls.MC_RESNET_LABEL_LEN = cls.MC_RESNET_INPUT_LEN // cls.MC_RESNET_PAT_LEN            #96
            cls.MC_RESNET_LABEL_SHAPE = (cls.MC_RESNET_LABEL_LEN, cls.MC_RESNET_LABEL_DIM)        #(96,128)
            
    @classmethod
    def setYoloParams(cls, gridH, gridW, gridCellH, gridCellW):
            cls.GRID_H, cls.GRID_W = gridH, gridW
            cls.GRID_CELL_H, cls.GRID_CELL_W = gridCellH, gridCellW
            cls.GRID_SIZE = cls.GRID_CELL_H * cls.GRID_CELL_W
            cls.NUM_PATTERNS = int(2 ** (cls.GRID_CELL_H * cls.GRID_CELL_W)) 
            cls.NUM_CLASSES = cls.NUM_PATTERNS #- 1 
            cls.YOLO_LABEL_DIM = cls.BOX_DIM + cls.NUM_CLASSES
            cls.YOLO_LABEL_SHAPE = (cls.GRID_H, cls.GRID_W, cls.NUM_BOXES, cls.YOLO_LABEL_DIM)
            cls.YOLO_OUTPUT_KERNAL_SIZE = cls.NUM_BOXES * cls.YOLO_LABEL_DIM
            cls.YOLO_NORM_H = cls.GRID_H * cls.GRID_CELL_H
            cls.YOLO_NORM_W = cls.GRID_W * cls.GRID_CELL_W
            cls.YOLO_BH = cls.GRID_CELL_H / cls.YOLO_NORM_H
            cls.YOLO_BW = cls.GRID_CELL_W / cls.YOLO_NORM_W
    
    @classmethod
    def setExprFeatures(cls, featureColumns, dsLabel):
        keyFeatureDict = {
            'Cross_BL_A_HH': ['AliSeq', ['Interface1']],
            'Cross_BL_P_HH': ['AliSeq', ['Interface1']],
            'Cross_HH_BL_P': ['sequence', ['p_interface']],
            'HH_Combined': ['AliSeq', ['Interface1']],
            'Homo_Hetro': ['AliSeq', ['Interface1']],
            'Homo': ['AliSeq', ['Interface1']],
            'Hetro': ['AliSeq', ['Interface1']],
            'Biolip_P': ['sequence', ['p_interface']],
            'Biolip_S': ['sequence', ['s_interface']],
            'Biolip_N': ['sequence', ['n_interface']],
            'Biolip_A': ['sequence', ['any_interface']],
            'Epitope': ['AliSeq', ['Interface1']],
            'COVID_P': ['sequence', ['p_interface']],
        }
        cls.EXPR_DATASET_LABEL = dsLabel
        keyFeatures = keyFeatureDict.get(cls.EXPR_DATASET_LABEL)
        cls.SEQ_COLUMN_NAME = keyFeatures[0]
        cls.LABEL_COLUMNS = keyFeatures[1]
        cls.FEATURE_COLUMNS = featureColumns
        cls.setCondParams()
        return
    
    @classmethod
    def getEncondingType(cls):
        return 'One-Hot' if cls.ONE_HOT_ENCODING else 'Prot2vec'
    
    @classmethod
    def getFeatureColumns(cls):
        return cls.FEATURE_COLUMNS.__str__()
    
    @classmethod
    def getFeaturesDim(cls):
        return cls.PROT_IMAGE_C

    
class PPIDatasetCls(object):
    oneHotDictionary = None
    protvecDictionary = None
    mcOneHotDictionary = None

    @classmethod
    def setLogger(cls, loggerParam):
        global logger
        logger = loggerParam
    
    @classmethod
    def augmentData(cls, seqList, sliceSize, newSliceShape):
        # newSliceShape[-1] is number of features or LABEL_DIM. We will shift one or more residu's; this means that we should shift one or more 
        # block of values belonging to the shifted residu's. That is going to be shiftSize.
        shiftSize = DatasetParams.SHIFT_WEIGHT * newSliceShape[-1]
        
        # We must limit shifting; otherwise repeated examples will be generated.
        numShifts = sliceSize // shiftSize
        
        allSlices = None
        for count in range(numShifts):
            slices = np.reshape(seqList, newSliceShape)
            if count != 0:
                # Remove the last peptide from the slices because it contains residu's from the the first peptide.
                slices = slices[:-1]
                allSlices = np.concatenate([allSlices, slices], axis=0)
            else:
                allSlices = slices                 
            seqList = np.roll(seqList, -shiftSize)
                
        #logger.info('$$$$allSlices.shape: %s', allSlices.shape)
        
        return allSlices 
    
    """
    seqList: contains all features of all residu's of a protein (sequence). There is no hierarchy and it's a flattend list.
    slices: is the conversion of the flat list into a number of slices (data examples) with a certain dimension.
    """
    @classmethod
    def sliceProt2Peptides(cls, seqList, sliceShape, padConst):
        # remember the first and last (which must be the number of features) dimensions.
        fdim = sliceShape[0]
        ldim = sliceShape[-1]
        
        # find the real length of a protein
        seqListLen = len(seqList)    
        protlen = seqListLen // ldim
        #print('protlen: ', protlen)
        
        if DatasetParams.USE_VAR_BATCH_INPUT:
            if fdim != None:    
                sys.exit("Error: The first dimension must be None (and the batch_size must be 1) if you want to use variable length input.")
            sliceShapeList = list(sliceShape)
            sliceShapeList[0] = protlen
            sliceShape = tuple(sliceShapeList)
            
        # We cut a protein to a number of slices (peptides). The linear size of a slice is equal to PROT_IMAGE_3D_SIZE. In this way, we are
        # able to produce more data (examples) in order to apply u-net. 
        sliceSize = 1
        for dim in sliceShape:
            sliceSize = sliceSize * dim
        
        # In case of features, sliceSize is "PROT_IMAGE_H * PROT_IMAGE_W * PROT_IMAGE_C";
        # but in case of labels sliceSize is "PROT_IMAGE_H * PROT_IMAGE_W * LABEL_DIM"    
        numPeptides = seqListLen // sliceSize
        protAndMaxLenEqual = (seqListLen == sliceSize)
        
        # numPeptides > 0 ===> protein-length is larger than the maximum length, and therefore this protein needs to be sliced. However, we slice it
        # if we are not instructed to to skip slicing. As a result, we remove this protein from the list. Note that the empty array must be reshaped,
        # otherwise numpy gives an error message. Because the shape of an empty array is (0,), the first dimension of the new shape of the reshape operation
        # must be zero. 
        if not protAndMaxLenEqual and numPeptides > 0 and DatasetParams.SKIP_SLICING:
            firstDim = (0,)
            newSliceShape = firstDim + sliceShape
            return np.array([], DatasetParams.FLOAT_TYPE).reshape(newSliceShape)
        
        remainder = seqListLen % sliceSize
        if remainder > 0: 
            paddingSize = sliceSize - remainder
            seqList = np.pad(seqList, (0,paddingSize), 'constant', constant_values=padConst)
            # We build a new slice (peptide) by padding the the remaining residu's.
            numPeptides += 1
        
        firstDim = (numPeptides,)
        newSliceShape = firstDim + sliceShape
        
        # Shape slices: (60, 8, 22) or (34, 8, 22) or ...
        if DatasetParams.USE_DATA_AUGMENTATION:
            allSlices = cls.augmentData(seqList, sliceSize, newSliceShape)
        else:
            allSlices = np.reshape(seqList, newSliceShape)
        # if variable length then allSlices.shape = (1, 159, 41) or (1, 88, 41) or (1, 80, 41), ... 
        #print(allSlices.shape)
        
        return allSlices 
    
    @classmethod
    def divideAllDataInBatches(cls, inputData, labelData, training):
        def divideDataInBatches(dataList, padConst):
            # dataList = [(num-of-prots,prot-len,num-features)] || [(num-of-prots,prot-len,num-labels)]
            # dataList = [(1, 159, 41), ..., (1, 88, 41)] || [(1, 159, 1), ..., (1, 88, 1)]
            batchesList = []
            minLen = 0
            for maxLen in DatasetParams.MAX_PROT_LEN_PER_BATCH: #e.g., [64,128,256,320,448,576,1024]
                # dataBatch = all prots which can be bundled in a batch, based on the condition minLen < protLen <= maxLen 
                dataBatch = [v for _,v in enumerate(dataList) if v.shape[1] > minLen and v.shape[1] <= maxLen]
                if dataBatch == []:
                    continue
                startingDBLoop = True
                for prot in dataBatch:
                    paddingSize = maxLen - prot.shape[1]
                    if paddingSize > 0: 
                        prot = np.pad(prot, ((0,0),(0,paddingSize),(0,0)), 'constant', constant_values=padConst)
                    if startingDBLoop:
                        paddedDataBatch = prot  
                        startingDBLoop = False
                    else:
                        paddedDataBatch = np.concatenate([paddedDataBatch, prot], axis=0)
                batchesList.append(paddedDataBatch)
                minLen = maxLen
            return batchesList
        
        doShuffel = True
        if training:
            trainX, valX, trainY, valY = train_test_split(inputData, labelData, test_size=DatasetParams.VALIDATION_RATE, shuffle=doShuffel) 
            DatasetParams.REAL_TRAIN_PROT_LENS = [v.shape[1] for _,v in enumerate(trainY)]
            if DatasetParams.MAX_PROT_LEN_PER_BATCH != None:    #means batch-size is 1 (variable length input)
                trainX = divideDataInBatches(trainX, DatasetParams.FEATURE_PAD_CONST)
                trainY = divideDataInBatches(trainY, DatasetParams.LABEL_PAD_CONST)  
            trainY = cls.encodeLabels(trainY, True)
        else:
            #we don't do any padding for the validation/test set.
            valX = inputData
            valY = labelData
            trainX, trainY = None, None
        DatasetParams.REAL_VAL_PROT_LENS = [v.shape[1] for _,v in enumerate(valY)]
        valY = cls.encodeLabels(valY, False)
        
        return trainX, valX, trainY, valY
        
    """
    slicesList: [array of shape (21, 4, 4, 1), ..., array of shape (12, 4, 4, 1)]
    concatenatedSlices: array of shape (5118, 4, 4, 1); the first dimensions have been concatenated.
    """
    @classmethod
    def concatSlices(cls, slicesList, labelData):
        # concat is not possible because we have different prolens: [(1, 159, 41), ..., (1, 88, 41)] 
        if DatasetParams.USE_VAR_BATCH_INPUT:
            #[(60, 43, 22), ..., (34, 1011, 22)]
            if labelData:
                logger.info("Length labelData: %s", len(slicesList))
            else:
                logger.info("Length inputData: %s", len(slicesList))
            return slicesList
        
        # slicesList: [(60, 8, 22), ..., (34, 8, 22)]
        concatenatedSlices = slicesList[0]
        for s in range(1, len(slicesList)):     
            slices = slicesList[s]
            concatenatedSlices = np.concatenate([concatenatedSlices, slices], axis=0)
        
        # Shape inputData: (10086, 8, 22)
        # Shape labelData: (10086, 8, 1)
        if labelData:
            logger.info("Shape labelData: %s", concatenatedSlices.shape)
        else:
            logger.info("Shape inputData: %s", concatenatedSlices.shape)
        
        return concatenatedSlices
        
    """
    - shape of featuresTable = (number of examples, number of features)
        - for the time being, number of features is 1 (only protein sequence) 
    - shape of inputData = (number of examples, PROT_IMAGE_H, PROT_IMAGE_W, PROT_IMAGE_C)
    """
    @classmethod
    def consInputs(cls, featuresTable, inputShape):
        # featuresRow = [    mean_H             RSA_q             AliSeq ]
        # featuresRow = ['0.23,...,0.45', '0.56,...,0.34', ..., 'A,...,T']
        def applyFeatureEncodingForEachInput(featuresRow):
            featureVals = []
            residuVal = []
            for featureInd in range(DatasetParams.FEATURES_SIZE):
                # type of featureStr is: String
                featureStr = featuresRow[featureInd]
                if featureInd == DatasetParams.SEQ_COLUMN_IND:
                    # Shape protEncodings for one example of protein-sequence length of 442: (44200,); 
                    # which is the concatenation of encodings for all residu's.
                    if DatasetParams.ONE_HOT_ENCODING:
                        residuVal = cls.getOneHotEncodings(featureStr)
                    else:    
                        residuVal = cls.getProtvecEncodings(featureStr) 
                      
                    # residuVal is a list of encodings for all residus: [c1R1,...,cnR1, c1R2,...,cnR2, ..., c1Rn,...,cnRn];
                    # we have to reshape it as: [ [c1R1,...,cnR1], 
                    #                             [c1R2,...,cnR2], 
                    #                             [     ...     ], 
                    #                             [c1Rn,...,cnRn] ]
                    numOfResidus = len(residuVal) // DatasetParams.PROT_RESIDU_ENCODING_DIM  
                    residuVal = np.array(residuVal).reshape((numOfResidus, DatasetParams.PROT_RESIDU_ENCODING_DIM))   
                    continue
                else:
                    # Value of (for example) the feature mean_H for all residues:      [mR1, mR2, ..., mRn],
                    # Value of (for example) the feature H_Entropies for all residues: [eR1, eR2, ..., eRn],
                    # Value of (for example) the feature AliSeq for all residues:      ['A', 'G', ..., 'V'],
                    #featureVal = list(eval(featureStr))
                    featureVal = np.fromstring(featureStr, dtype=DatasetParams.FLOAT_TYPE, sep=',')
                    #featureVal = featureVal.tolist()
                    #print(len(featureVal))
                featureVals.append(featureVal)
            featureVals = np.array(featureVals)  
            #logger.info("Shape featureVals: %s", featureVals.shape)
            #print(featureVals)
            
            # We convert the above feature-based matrix to a residu-based matrix: [mR1, eR1, ..., 'A'],
            #                                                                     [mR2, eR2, ..., 'G'],
            #                                                                     [          ...     ], 
            #                                                                     [mRn, eRn, ..., 'V']
            # and concatenate it with the residu-based for encodings.
            featureVals = np.transpose(featureVals) 
            
            # We have to deal with the following situations where featureVals or residuVal is empty: (1) no features except for AliSeq, (2) no AliSeq;
            # otherwise we get numpy.concatenate error.
            m = featureVals.shape[0]
            if m == 0:
                featureVals = featureVals.reshape(numOfResidus,-1)
            else:
                residuVal = np.array(residuVal).reshape((m,-1))    
            
            #nonlocal proteinCount
            #proteinCount += 1
            #logger.info("Protein count: %s || Protein length: %s", proteinCount, numOfResidus)
                
            #logger.info("Shape featureVals: %s", featureVals.shape)
            featureVals = np.concatenate([featureVals, residuVal], axis=1)   
            #logger.info("Shape featureVals: %s", featureVals.shape)
            featureVals = featureVals.flatten()
            # Shape featureVals: (15480,)
            #logger.info("Shape featureVals: %s", featureVals.shape)
            featureVals = cls.sliceProt2Peptides(featureVals, inputShape, DatasetParams.FEATURE_PAD_CONST)
    
            # Shape featureVals: (42, 8, 22) or (34, 8, 22) or ...
            #logger.info("Shape featureVals: %s", featureVals.shape)
            
            return featureVals
    
        proteinCount = 0
        inputDataList = list(map(applyFeatureEncodingForEachInput, featuresTable))
        inputData = cls.concatSlices(inputDataList, labelData=False)
        #print(featuresTable[0,:])
        #print(inputData[0, :, :, :])
        
        return inputData

    @classmethod
    def encodeMCResnetLabels(cls, labelData, isTraining):
        def generatePatterns():
            classLen = DatasetParams.MC_RESNET_PAT_LEN
            # Generate all possible binary matrixes of size: classLen x 1.
            patterns = np.asarray([np.reshape(np.array(i), (classLen, 1)) 
                                   for i in it.product([0, 1], repeat = classLen * 1)])
            k,h,w = patterns.shape                  # (128, 7, 1)
            patterns = np.reshape(patterns, (k,h))  # (128, 7)
            return patterns
        
        def getPatClass(gridCell):
            #print(gridCell.tolist())
            bb = np.zeros((DatasetParams.MC_RESNET_LABEL_DIM,), DatasetParams.FLOAT_TYPE)
            classId = DatasetParams.MC_RESNET_PATTERNS.tolist().index(gridCell.tolist())
            #print(classId)
            bb[0:DatasetParams.MC_RESNET_LABEL_DIM] = cls.getMCOneHotEncoding(classId)
            #print(bb)
            return bb
        
        if isTraining:
            realProtLens = DatasetParams.REAL_TRAIN_PROT_LENS
        else:
            realProtLens = DatasetParams.REAL_VAL_PROT_LENS
            
        DatasetParams.MC_RESNET_PATTERNS = generatePatterns()
        n,h,l = labelData.shape     # (278, 768, 1)
        firstDim = (n,)
        newAttLabelShape = firstDim + DatasetParams.MC_RESNET_LABEL_SHAPE 
        attLabels = np.zeros(newAttLabelShape, DatasetParams.FLOAT_TYPE)
        #print(attLabels.shape)      # (278, 130, 128)
        
        tmpLabelShape = (n, DatasetParams.MC_RESNET_LABEL_LEN, DatasetParams.MC_RESNET_PAT_LEN)   # (278, 128, 6)
        labelData = labelData.reshape(tmpLabelShape)
        for i in range(n):
            protLen = realProtLens[i]
            protLabelLen = protLen // DatasetParams.MC_RESNET_PAT_LEN
            if protLen % DatasetParams.MC_RESNET_PAT_LEN != 0:
                protLabelLen = protLabelLen + 1
            for j in range(protLabelLen): 
                gridCell = labelData[i,j]
                attLabels[i,j] = getPatClass(gridCell)
        #print(attLabels[0,0])
        #print(attLabels[0,1])
        #print(attLabels[0,10])
        return attLabels
    
    @classmethod
    def encodeAttLabels(cls, labelData, isTraining):
        def generatePatterns():
            # We have to increment ATT_PAT_LEN by one bit to make place for ATT_DECODER_START and ATT_DECODER_END.  
            attClassLen = DatasetParams.ATT_PAT_LEN + 1
            # Generate all possible binary matrixes of size: attClassLen x 1.
            patterns = np.asarray([np.reshape(np.array(i), (attClassLen, 1)) 
                                   for i in it.product([0, 1], repeat = attClassLen * 1)])
            k,h,w = patterns.shape                  # (128, 7, 1)
            patterns = np.reshape(patterns, (k,h))  # (128, 7)
            return patterns
        
        def isZeroPattern(gridCell):
            return np.count_nonzero(gridCell) == 0
        
        def getPatClass(gridCell):
            #print(gridCell.tolist())
            bb = np.zeros((DatasetParams.ATT_LABEL_DIM,), DatasetParams.FLOAT_TYPE)
            classId = DatasetParams.ATT_PATTERNS.tolist().index(gridCell.tolist())
            #print(classId)
            bb[0:DatasetParams.ATT_LABEL_DIM] = cls.getMCOneHotEncoding(classId)
            #print(bb)
            return bb
        
        if isTraining:
            realProtLens = DatasetParams.REAL_TRAIN_PROT_LENS
        else:
            realProtLens = DatasetParams.REAL_VAL_PROT_LENS
            
        DatasetParams.ATT_PATTERNS = generatePatterns()
        n,h,l = labelData.shape     # (278, 768, 1)
        firstDim = (n,)
        newAttLabelShape = firstDim + DatasetParams.ATT_EXT_LABEL_SHAPE 
        attLabels = np.zeros(newAttLabelShape, DatasetParams.FLOAT_TYPE)
        #print(attLabels.shape)      # (278, 130, 128)
        
        # We use the one-hot-encoding of [1,1,1,1,1,1,0] and [1,1,1,1,1,1,1], which are the last two elements of the patterns, as the START/END-of-sentence.
        DatasetParams.ATT_DECODER_START = getPatClass(DatasetParams.ATT_PATTERNS[2 ** DatasetParams.ATT_PAT_LEN])    # [1,0,0,0,0,0,0]=64 | (128,)
        DatasetParams.ATT_DECODER_END = getPatClass(DatasetParams.ATT_PATTERNS[DatasetParams.ATT_LABEL_DIM-1])       # [1,1,1,1,1,1,1]=127 | (128,)
        
        tmpLabelShape = (n, DatasetParams.ATT_LABEL_LEN, DatasetParams.ATT_PAT_LEN)   # (278, 128, 6)
        labelData = labelData.reshape(tmpLabelShape)
        for i in range(n):
            attLabels[i,0] = DatasetParams.ATT_DECODER_START
            protLen = realProtLens[i]
            protLabelLen = protLen // DatasetParams.ATT_PAT_LEN
            if protLen % DatasetParams.ATT_PAT_LEN != 0:
                protLabelLen = protLabelLen + 1
            # we have already used attLabels[i,0], therefore the index must be protLabelLen+1.
            attLabels[i,protLabelLen+1] = DatasetParams.ATT_DECODER_END    
            for j in range(protLabelLen): 
                gridCell = labelData[i,j]
                # Extend gridCell; add additional zero at the beginning; [1,0,0,1,1,0] becomes [0,1,0,0,1,1,0].
                gridCell = np.insert(gridCell, 0, 0)
                #if isZeroPattern(gridCell): # this is a bug!!
                #    continue
                # we have already inserted the START at position zero, therefore j+1.
                attLabels[i,j+1] = getPatClass(gridCell)
        #print(attLabels[0,0])
        #print(attLabels[0,1])
        #print(attLabels[0,10])
        return attLabels
    
    @classmethod
    def encodeYoloLabels(cls, labelData):
        def generatePatterns():
            # Generate all possible binary matrixes of size: GRID_CELL_H x GRID_CELL_W.
            patterns = np.asarray([np.reshape(np.array(i), (DatasetParams.GRID_CELL_H, DatasetParams.GRID_CELL_W)) 
                                   for i in it.product([0, 1], repeat = DatasetParams.GRID_CELL_H * DatasetParams.GRID_CELL_W)])
            # (512, 3, 3)
            #print(patterns.shape)
            #print(patterns)
            # First pattern
            #print(patterns[10,:,:])
            # compare pattern with a given pattern
            #print(np.array_equal(patterns[1,:,:], [[0,0,0],[0,0,0],[0,0,1]]))
            
            return patterns
        
        def isZeroPattern(gridCell):
            return np.count_nonzero(gridCell) == 0
        
        def getBBCharacteristics(gridCell):
            ld = np.count_nonzero(np.diag(gridCell))
            rd = np.count_nonzero(np.diag(np.fliplr(gridCell)))
            hl = np.count_nonzero(gridCell[0,:])
            hh = np.count_nonzero(gridCell[-1,:])
          
            return np.array([ld/DatasetParams.GRID_SIZE, rd/DatasetParams.GRID_SIZE, hl/DatasetParams.GRID_SIZE, hh/DatasetParams.GRID_SIZE])
        
        def getBBPositions(h, w):
            """
            #box = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]
            #[h,w]=[3,1]: xmin=w*gch=2,xmax=xmin+(gch-1)=3,ymin=h*gcw=6,ymax=ymin+(gcw-1)=7
            #[h,w]=[0,3]: xmin=w*gch=6,xmax=xmin+(gch-1)=7,ymin=h*gcw=0,ymax=ymin+(gcw-1)=1
            
            #bx = (.5 * (xmin + xmax)) / DatasetParams.GRID_CELL_W
            #by = (.5 * (ymin + ymax)) / DatasetParams.GRID_CELL_H
            #bh = (xmax - xmin) / DatasetParams.GRID_CELL_W
            #bw = (ymax - ymin) / DatasetParams.GRID_CELL_H
            
            bx = DatasetParams.GRID_CELL_W // 2
            by = DatasetParams.GRID_CELL_H // 2
            bh = DatasetParams.YOLO_BH
            bw = DatasetParams.YOLO_BW
            
            bx = randint(1, DatasetParams.GRID_CELL_W) // DatasetParams.GRID_CELL_W 
            by = randint(1, DatasetParams.GRID_CELL_H) // DatasetParams.GRID_CELL_H
            bh = DatasetParams.YOLO_BH
            bw = DatasetParams.YOLO_BW
           
            gx = w * DatasetParams.GRID_CELL_W
            gy = h * DatasetParams.GRID_CELL_H
            ox = gx + (DatasetParams.GRID_CELL_W // 2)
            oy = gy + (DatasetParams.GRID_CELL_H // 2)
            try:
                bx = (ox - gx) / gx
                by = (oy - gy) / gy
            except ZeroDivisionError:
                bx = (ox - gx)
                by = (oy - gy)
            bh = DatasetParams.YOLO_BH
            bw = DatasetParams.YOLO_BW
            """
            xmin = w * DatasetParams.GRID_CELL_W
            xmax = xmin + (DatasetParams.GRID_CELL_W - 1)
            ymin = h * DatasetParams.GRID_CELL_H
            ymax = ymin + (DatasetParams.GRID_CELL_H - 1)
            bx, by, bh, bw = xmin, ymin, xmax, ymax
             
            
            return np.array([bx, by, bh, bw])
        
        def getBoundingBox(gridCell, h, w):
            #print(gridCell.tolist())
            bb = np.zeros((DatasetParams.YOLO_LABEL_DIM,), DatasetParams.FLOAT_TYPE)
            bb[0:4] = getBBPositions(h, w)
            #bb[0:4] = getBBCharacteristics(gridCell)
            bb[4] = 1.0
            classId = DatasetParams.YOLO_PATTERNS.tolist().index(gridCell.tolist())
            #print(classId)
            bb[5:DatasetParams.YOLO_LABEL_DIM] = cls.getMCOneHotEncoding(classId)
            #print(bb)
            
            return bb
        
        def consGrids(labelData, tmpLabelShape):
            n,gridH,gridW,gridCellH,gridCellW = tmpLabelShape 
            labelData = labelData.reshape(n, gridH, gridCellH, gridW, gridCellW).swapaxes(2, 3).reshape(tmpLabelShape)
            
            return labelData
        
        DatasetParams.YOLO_PATTERNS = generatePatterns()
        # (278, 24, 24, 1)
        n,h,w,l = labelData.shape
        # (278, 12, 12, 1, 21)
        firstDim = (n,)
        newYoloLabelShape = firstDim + DatasetParams.YOLO_LABEL_SHAPE
        yoloLabels = np.zeros(newYoloLabelShape, DatasetParams.FLOAT_TYPE)
        #print(yoloLabels.shape)
        
        # (278, 12, 12, 2, 2)
        tmpLabelShape = (n, DatasetParams.GRID_H, DatasetParams.GRID_W, DatasetParams.GRID_CELL_H, DatasetParams.GRID_CELL_W)
        # convert labelData to grids
        labelData = consGrids(labelData, tmpLabelShape)
        for i in range(n):
            for j in range(DatasetParams.GRID_H):
                for k in range(DatasetParams.GRID_W):
                    gridCell = labelData[i,j,k]
                    if isZeroPattern(gridCell):
                        continue
                    for b in range(DatasetParams.NUM_BOXES):
                        yoloLabels[i,j,k,b,:] = getBoundingBox(gridCell, j, k)
        
        return yoloLabels
    
    """
    - shape of labelVector = (number of examples, )
    - we have to determine per grid the type of pattern fitting in the grid.
    """
    @classmethod
    def consLabels(cls, labelvec, labelShape):
        def applyLabelingForEachInput(labelStr):
            # type of labelStr is: narray of one element which is a String.
            # Convert '1,0,0,...1,0,1' to [1, 0, ..., 0, 1].
            #realLabels = list(eval(labelStr[0]))
            #realLabels = list(np.float32(realLabels))
            realLabels = np.fromstring(labelStr[0], dtype=DatasetParams.FLOAT_TYPE, sep=',')
            #realLabels = realLabels.tolist()
            realLabels = cls.sliceProt2Peptides(realLabels, labelShape, padConst)
            #logger.info("Shape realLabels: %s", realLabels.shape)
            return realLabels
        
        # We will use this to extract the non-padded part (the real protein without padding) 
        padConst = DatasetParams.LABEL_PAD_CONST
        
        labelDataList = list(map(applyLabelingForEachInput, labelvec))
        labelData = cls.concatSlices(labelDataList, labelData=True)
        #print(labelData[0,:,:])

        return labelData
    
    @classmethod
    def encodeLabels(cls, labelData, isTraining):
        if DatasetParams.USE_ATT:
            labelData = cls.encodeAttLabels(labelData, isTraining)
            # (278, 96, 64)
            logger.info("Shape attLabel: %s", labelData.shape)
            #print(labelData[0,0])
        elif DatasetParams.USE_MC_RESNET:
            labelData = cls.encodeMCResnetLabels(labelData, isTraining)
            # (278, 96, 64)
            logger.info("Shape mcResnetLabel: %s", labelData.shape)
            #print(labelData[0,0])
        elif DatasetParams.USE_YOLO:
            labelData = cls.encodeYoloLabels(labelData)
            # (278, 12, 12, 1, 20)
            logger.info("Shape yoloLabel: %s", labelData.shape)
            #print(labelData[0,:,:,:])
            
        return labelData
    
    @classmethod
    def performZeroDownSampling(cls, inputData, labelData, numOnes):
        downSamplingThreshold = int(DatasetParams.DOWN_SAMPLING_THRESHOLD * np.mean(numOnes))
        # collect indeces of peptides which doesn't have an interface at all.
        zeroInds = np.where(numOnes <= downSamplingThreshold)[0]
        #print('zeroInds[0..10]: ', zeroInds[0:9], ' | zeroInds.shape: ', zeroInds.shape, ' | zeroInds.size: ', zeroInds.size)
        if zeroInds.size > 0:   # this means there are peptides having no interfaces.
            oneInds = np.where(numOnes > downSamplingThreshold)[0]
            lenZeros = zeroInds.size
            lenOnes = oneInds.size
            #print('lenZeros: ', lenZeros, ' | lenOnes: ', lenOnes)
            downSize = abs(lenZeros - lenOnes)
            #print('downSize: ', downSize)
            if downSize > 0:
                downInds = zeroInds if lenZeros > lenOnes else oneInds
                downInds = np.random.choice(downInds, size=downSize, replace=False)
                numOnes = np.delete(numOnes, downInds, axis=0)
                labelData = np.delete(labelData, downInds, axis=0)
                inputData = np.delete(inputData, downInds, axis=0)
        
        logger.info("Shape inputData (zero down sampled): %s", inputData.shape)
        logger.info("Shape labelData (zero down sampled): %s", labelData.shape)
        
        return inputData, labelData, numOnes
    
    @classmethod
    def performZscoreDownSampling(cls, inputData, labelData, numOnes):
        numProts = len(numOnes)
        # convert numOnes (the set representing interface status (number of 1's) per peptide) to z-score in order to remove outliers.
        zNumOnes = stats.zscore(numOnes, axis=0)
        #print('znums[0..10]: ', zNumOnes[0:9])
        stdNum = DatasetParams.DOWN_SAMPLING_STD
        tobeDeletedInds = []
        for i in range (numProts):
            if np.abs(zNumOnes[i]) > stdNum:
                tobeDeletedInds.append(i)
        labelData = np.delete(labelData, tobeDeletedInds, axis=0)
        inputData = np.delete(inputData, tobeDeletedInds, axis=0)
        
        logger.info("Shape inputData (Zscore down sampled): %s", inputData.shape)
        logger.info("Shape labelData (Zscore down sampled): %s", labelData.shape)
    
        return inputData, labelData
    
    @classmethod
    def performDownSampling(cls, inputData, labelData):
        try:
            numProts = labelData.shape[0]
        except:
            sys.exit("Error: Using USE_DOWN_SAMPLING together with USE_VAR_BATCH_INPUT is not allowed.")
            
        numOnes = np.zeros(numProts)
        #print('labelData[0]: ', labelData[0,...,-1])
        
        # collect, per peptide, number of interfaces (1's): [3,9,0,0,...,3]
        for i in range (numProts):
            label = labelData[i]
            #numOnes[i] = np.count_nonzero(labelData[i,...,-1])
            numOnes[i] = np.count_nonzero(label[...,-1]==1)
        #print('numones[0..10]: ', numOnes[0:9])
        logger.info("sum of numones: {:02d}".format(int(np.sum(numOnes))))
        
        inputData, labelData, numOnes = cls.performZeroDownSampling(inputData, labelData, numOnes)
        inputData, labelData = cls.performZscoreDownSampling(inputData, labelData, numOnes)

        return inputData, labelData
    
    @classmethod    
    def splitData(cls, inputData, labelData):
        doShuffel = True    
        randState = 42 #None
        testSize = DatasetParams.VALIDATION_RATE #100
        stratify = None
        trainX, valX, trainY, valY = train_test_split(inputData, labelData, test_size=testSize, random_state=randState, shuffle=doShuffel, stratify=stratify)
        return trainX, valX, trainY, valY
    
    @classmethod    
    def averageData(cls, data):
        # data.shape=(17930, 1, 4)
        data = np.mean(data, axis=2)
        data = np.reshape(data, (data.shape[0],data.shape[1],1))
        # data.shape=(17930, 1, 1)
        return data
    
    @classmethod    
    def sliceData(cls, data, sliceShape, isLabel):
        if isLabel:
            padConst = DatasetParams.LABEL_PAD_CONST
        else:
            padConst = DatasetParams.FEATURE_PAD_CONST
        data = K.flatten(data)
        slicedData = cls.sliceProt2Peptides(data, sliceShape, padConst)
        return slicedData
    
    @classmethod
    def getRealProtLens(cls, dataset):
        protLens = dataset[DatasetParams.REAL_LEN_COL_NAME]
        return protLens.tolist()
    
    @classmethod
    def getProtIds(cls, dataset):
        protIds = dataset[DatasetParams.PROT_ID_NAME]
        return protIds.tolist()
    
    @classmethod    
    def makeDataset(cls, dataFile, inputShape, labelShape, training, partioning=False):
        dataset = pd.read_csv(dataFile)
        if partioning:
            dataset = dataset.sample(frac=DatasetParams.VALIDATION_RATE,random_state=DatasetParams.RANDOM_STATE)
        #DatasetParams.PROT_IDS = cls.getProtIds(dataset)
        featuresTable = dataset.loc[:, DatasetParams.FEATURE_COLUMNS].values.astype(np.str)
        labelvec = dataset.loc[:, DatasetParams.LABEL_COLUMNS].values.astype(np.str)
        # Shape featuresTable: (288, 20) | Shape labelVector: (288, 1)
        logger.info("Shape featuresTable: %s | Shape labelvec: %s", featuresTable.shape, labelvec.shape)
        
        #Shape inputData: (288, 32, 32, 20)
        #Shape labelData: (288, 32, 32, 1)
        inputData = cls.consInputs(featuresTable, inputShape)
        labelData = cls.consLabels(labelvec, labelShape)
        
        if DatasetParams.USE_DOWN_SAMPLING:
            inputData, labelData = cls.performDownSampling(inputData, labelData)
        
        if not DatasetParams.USE_VAR_BATCH_INPUT:
            if training:
                trainX, valX, trainY, valY = cls.splitData(inputData, labelData)
                trainY = cls.encodeLabels(trainY, True)
                #print('88888888888 labelData: ', labelData.shape, ' | trainY: ', trainY.shape,  ' | valY: ', valY.shape )
            else:
                valX = inputData
                valY = labelData
                trainX, trainY = None, None
            DatasetParams.REAL_VAL_PROT_LENS = cls.getRealProtLens(dataset)
            valY = cls.encodeLabels(valY, False)
        else:
            trainX, valX, trainY, valY = cls.divideAllDataInBatches(inputData, labelData, training)
        
        return trainX, valX, trainY, valY 
    
    @classmethod
    def createMCOneHotDict(cls):
        classes = list(range(DatasetParams.NUM_CLASSES))
        num2classes = {aa:idx for idx, aa in enumerate(classes)}
        ohIdx = list(num2classes.values())
        oh = U.to_categorical(ohIdx)
        i = 0
        cls.mcOneHotDictionary = num2classes.copy()
        for key in cls.mcOneHotDictionary.keys():
            cls.mcOneHotDictionary[key] = list(oh[i])
            i = i + 1
        #print(cls.mcOneHotDictionary)
        
        return cls.mcOneHotDictionary
    
    @classmethod
    def getMCOneHotEncoding(cls, classId):
        if cls.mcOneHotDictionary == None:
            cls.mcOneHotDictionary = cls.createMCOneHotDict()
    
        try:
            ohEncoding = cls.mcOneHotDictionary[classId]
        except KeyError:
            logger.error('@@@ No one-hot-encoding for the claasId: %s', classId)
            sys.exit(1)
        ohEncoding = np.array(ohEncoding)    
        #print(ohEncoding)
        
        return ohEncoding
        
    @classmethod
    def createOneHotDict(cls):
        aas = DatasetParams.AA_LIST
        num2aa = {aa:idx for idx, aa in enumerate(aas)}
        ohIdx = list(num2aa.values())
        oh = U.to_categorical(ohIdx)
        i = 0
        cls.oneHotDictionary = num2aa.copy()
        for key in cls.oneHotDictionary.keys():
            cls.oneHotDictionary[key] = list(oh[i])
            i = i + 1
        #print(cls.oneHotDictionary)
        
        return cls.oneHotDictionary
    
    """
    seqStr: a string containing comma separated residue-characters; "A,A,S,...,G,F" 
    """
    @classmethod
    def getOneHotEncodings(cls, seqStr):
        if cls.oneHotDictionary == None:
            cls.oneHotDictionary = cls.createOneHotDict()
        
        # Convert "A,A,S,...,G,F" to ['A', 'A', 'S' ..., 'G', 'F'].
        aaList = np.array(seqStr.split(','))
        ohEncodings = []
        for aa in aaList:
            try:
                ohEncoding = cls.oneHotDictionary[aa]
            except KeyError:
                ohEncoding = cls.oneHotDictionary[DatasetParams.UNKNOWN_AA]
            ohEncodings = np.concatenate([ohEncodings, ohEncoding])
        #print(ohEncodings)
        
        return ohEncodings  
    
    @classmethod    
    def createProtvecDict(cls):
        protvec = np.genfromtxt(DatasetParams.PROT_VEC_ENCODINGS_FILE, delimiter='\t')
        # Remove aa names.
        protvec = protvec[:, 1:].astype(DatasetParams.FLOAT_TYPE) 
        cls.protvecDictionary = {}
        with open(DatasetParams.PROT_VEC_ENCODINGS_FILE, 'r') as f:
            triplets = [l.split('\t')[0] for l in f.readlines()]
    
        for idx, triplet in enumerate(triplets):
            cls.protvecDictionary[triplet] = protvec[idx]
    
        return cls.protvecDictionary  
    
    """
    seqStr = 'E,H,T,K,M'
    return = ['XEH', 'EHT', 'HTK', 'TKM', 'KMX']
    """
    @classmethod
    def seq2triplets(cls, seqStr):
        seqList = seqStr.split(',')
        # pad sequence with two 'X' to have correct number of triplets
        seqListLen = len(seqList)
        paddedSeq = ['X'] + seqList + ['X']
        triplets = []
        for i in range(1, seqListLen+1):
            triplets += ["".join(paddedSeq[i-1:i+2])]
            
        return triplets 
    
    """
    seqStr = 'E,H,T,K,M'
    return = [-0.134299, 0.64757901, -0.83385801, -0.257866, ..., -0.045017]
    length of the return-list is len(seqStr) * 100 
    """
    @classmethod
    def getProtvecEncodings(cls, seqStr):
        if cls.protvecDictionary == None:
            cls.protvecDictionary = cls.createProtvecDict()
            
        triplets = cls.seq2triplets(seqStr)
        #print(triplets)
        protvecEncodings = []
        for tripletPos in range(len(triplets)):
            triplet = triplets[tripletPos]
            #print(triplet)
            try:
                protvecEncoding = cls.protvecDictionary[triplet]
            except KeyError:
                protvecEncoding = cls.protvecDictionary[DatasetParams.UNKNOWN_TRIPLET]
            protvecEncodings = np.concatenate([protvecEncodings, protvecEncoding])
    
        return protvecEncodings  
    
    @classmethod
    def testProtvecEncoding(cls):
        print("============= testing Prot2Vec Encoding ============")
        seqStr = 'A,A,S,S,L,D'
        protEncodings = cls.getProtvecEncodings(seqStr)
        print(protEncodings.shape)
        print(protEncodings)
    
    @classmethod    
    def testOneHotEncoding(cls):
        print("============= testing OneHot Encoding ============")
        seqStr = 'A,A,S,S,L,D'
        protEncodings = cls.getOneHotEncodings(seqStr)
        print(protEncodings.shape)
        print(protEncodings) 
    
class DatasetPreparation(object):
    @classmethod
    def getAALabel(cls, aa1LetCode):
        aa1To3Dict = {
            'A': ['Alanine', 'ala'],
            'R': ['Arginine', 'arg'],
            'N': ['Asparagine', 'asn'],
            'D': ['Aspartate', 'asp'],
            #'B': ['Aspartate or Asparagine', 'Asx'],
            'C': ['Cysteine', 'cys'],
            'E': ['Glutamate', 'glu'],
            'Q': ['Glutamine', 'gln'],
            #'Z': ['Glutamate or Glutamine', 'Glx'],
            'G': ['Glycine', 'gly'],
            'H': ['Histidine', 'his'],
            'I': ['Isoleucine', 'ile'],
            'L': ['Leucine', 'leu'],
            'K': ['Lysine', 'lys'],
            'M': ['Methionine', 'met'],
            'F': ['Phenylalanine', 'phe'],
            'P': ['Proline', 'pro'],
            'S': ['Serine', 'ser'],
            'T': ['Threonine', 'thr'],
            'W': ['Tryptophan', 'trp'],
            'Y': ['Tyrosine', 'tyr'],
            'V': ['Valine', 'val'],
        }
        if aa1LetCode in aa1To3Dict.keys():
            return aa1To3Dict.get(aa1LetCode)
        
        return ['X','X']
    
    @classmethod
    def getStatLabel(cls, fileName):
        # if a key in this dictionary occurs in the file name of a dataset we prefix a statistics metric (e.g. len_freq) by
        # the value of that key.
        statLabelDict = {
            'training': 'train_',
            'testing': 'test_',
            'benchmark': 'bench_',
        } 
        for key in statLabelDict.keys():
            if key in fileName:
                return statLabelDict.get(key)
        print('%%Error - None of keys of statLabelDict occurs in the given file-name: ' + fileName)
        return ''

    @classmethod
    def getDatasetLabel(cls, fileName):
        datasetLabelDict = {
            'win_n': 'Biolip_N',
            'win_s': 'Biolip_S',
            'win_p': 'Biolip_P',
            'win_a': 'Biolip_A',
            'combined': 'Homo_Hetro',
            'homo': 'Homo',
            'hetro': 'Hetro',
            'epitope': 'Epitope',
        } 
        for key in datasetLabelDict.keys():
            if key in fileName:
                return datasetLabelDict.get(key)
        print('%%Error - None of keys of datasetLabelDict occurs in the given file-name: ' + fileName)
        return ''
    
    @classmethod
    def getStatFileName(cls, fileName):
        statFileDict = {
            'win_n': DatasetParams.PREPARED_BIOLIP_WIN_N_TRAINING_FILE.replace('prepared_', 'Stat_'),
            'win_s': DatasetParams.PREPARED_BIOLIP_WIN_S_TRAINING_FILE.replace('prepared_', 'Stat_'),
            'win_p': DatasetParams.PREPARED_BIOLIP_WIN_P_TRAINING_FILE.replace('prepared_', 'Stat_'),
            'win_a': DatasetParams.PREPARED_BIOLIP_WIN_A_TRAINING_FILE.replace('prepared_', 'Stat_'),
            'combined': DatasetParams.PREPARED_COMBINED_TRAINING_FILE.replace('prepared_', 'Stat_'),
            'homo': DatasetParams.PREPARED_COMBINED_TRAINING_FILE.replace('prepared_', 'Stat_'),
            'hetro': DatasetParams.PREPARED_COMBINED_TRAINING_FILE.replace('prepared_', 'Stat_'),
        } 
        for key in statFileDict.keys():
            if key in fileName:
                return statFileDict.get(key)
        print('%%Error - None of keys of statFileDict occurs in the given file-name: ' + fileName)
        return ''
    
    @classmethod
    def initExperiment(cls, projName, exprName):
        DatasetParams.COMET_EXPERIMENT = None
        DatasetParams.COMET_EXPERIMENT = Experiment(project_name=projName)
        DatasetParams.COMET_EXPERIMENT.disable_mp()
        DatasetParams.COMET_EXPERIMENT.set_name(exprName)
        return
    
    @classmethod
    def uploadExprFigures(cls, projName):
        cls.initExperiment(projName, 'stat-figures')
        DatasetParams.COMET_EXPERIMENT.log_asset_folder('../data/figures')
        return
    
    @classmethod
    def uploadExprHtmls(cls, projName):
        cls.initExperiment(projName, 'stat-htmls')
        DatasetParams.COMET_EXPERIMENT.log_asset_folder('../data/htmls')
        return
    
    @classmethod
    def normalizeData1(cls, df, colNames):
        df.fillna(DatasetParams.MISSING_VALUE_CONST, inplace=True)
        for colName in colNames:
            try:
                minVal = df[colName].min()
                maxVal = df[colName].max()
                # if dataset has only one protein then all residues have the same length; as a result we get NaNs. So here we put
                # minLen and maxLen of proteins.
                if minVal == maxVal:  
                    minVal = 20
                    maxVal = 35000
                #print("colName={} min={} and max={}".format(colName, minVal, maxVal))
                df[colName] = (df[colName] - minVal) / (maxVal - minVal)
                #print('value: ', df[colName])
                #print('==========================================')
            except KeyError:
                pass
        return df
    
    @classmethod
    def normalizeData(cls, df, colNames):
        df.fillna(DatasetParams.MISSING_VALUE_CONST, inplace=True)
        for colName in colNames:
            try:
                if colName == DatasetParams.LEN_COL_NAME:
                    minVal = 20 #50
                    maxVal = 2050 #10000 #35000
                else:
                    minVal = df[colName].min()
                    maxVal = df[colName].max()
                #print("colName={} min={} and max={}".format(colName, minVal, maxVal))
                df[colName] = (df[colName] - minVal) / (maxVal - minVal)
            except KeyError:
                pass
        return df
    
    @classmethod
    def normalizeData2(cls, df, colNames):
        df.fillna(DatasetParams.MISSING_VALUE_CONST, inplace=True)
        for colName in colNames:
            try:
                minVal = df[colName].min()
                maxVal = df[colName].max()
                #print("colName={} min={} and max={}".format(colName, minVal, maxVal))
                df[colName] = (df[colName] - minVal) / (maxVal - minVal)
            except KeyError:
                pass
        return df
    
    @classmethod
    def normalizeData3(cls, df, colNames):
        # both better performance and faster convergence.
        df.fillna(DatasetParams.MISSING_VALUE_CONST, inplace=True)
        for colName in colNames:
            try:
                avgVal = df[colName].mean()
                stdVal = df[colName].std()
                df[colName] = (df[colName] - avgVal) / stdVal
            except KeyError:
                pass
        return df
    
    @classmethod
    def convertLabels(cls, df, fieldName):
        labelMap = {fieldName:     {DatasetParams.LABELS[0]: 1, DatasetParams.LABELS[1]: 0}}
        df.replace(labelMap, inplace=True)
        return df
      
    @classmethod
    def getProtLens(cls, dataset, lenColName=DatasetParams.LEN_COL_NAME):
        lenCol = dataset[lenColName]
        protIndex = 0
        protLens = []
        try:
            while True:
                protLen = lenCol[protIndex]
                protIndex += protLen
                protLens.append(protLen)
        except:
            pass
        return protLens
    
    @classmethod
    def getProtIds(cls, dataset, lenColName=DatasetParams.LEN_COL_NAME):
        lenCol = dataset[lenColName]
        try:
            idCol = dataset[DatasetParams.PROT_ID_NAME]
        except: #dataset doesn't contain uniprot_id
            protIds = ['prot-'+str(i) for i in range(lenCol.size)]
            return protIds
        
        protIndex = 0
        protIds = []
        try:
            while True:
                protLen = lenCol[protIndex]
                protId = idCol[protIndex]
                protIndex += protLen
                protIds.append(protId)
        except:
            pass
                
        return protIds
    
    @classmethod
    def prepareData(cls, origFileName, preparedFileName=None, labelConv=True, includeId=False):
        if preparedFileName != None:
            print("\n## Generating prepared data for @@: " + preparedFileName + " @@")
            
        dataset = pd.read_csv(origFileName)
        if labelConv:
            dataset = cls.convertLabels(dataset, DatasetParams.LABEL_COL_NAME)
        protLens = cls.getProtLens(dataset)
        #includeId = False
        if includeId:
            protIds = cls.getProtIds(dataset)
        dataset = cls.normalizeData(dataset, DatasetParams.NORMALIZABLE_COLUMNS)
        
        print('Number of proteins: ', len(protLens))    #homo: 288     hetro: 119
        print('Max length of proteins: ', max(protLens))    #homo: 1010    hetro: 766
        print('Min length of proteins: ', min(protLens))    #homo: 66      hetro: 39
        print('Length of proteins: ', protLens)         #homo: []      hetro: []
        
        # Note that if you add LABEL_COL_NAME into ALL_COLUMNS then you have to remove it from the list of columns, defined as constant. 
        allColumns = DatasetParams.ALL_COLUMNS + [DatasetParams.LABEL_COL_NAME] 
        # REAL_LEN_COL_NAME wil have just one value (and not a list of values); therefore we exclude it from ALL_COLUMNS. 
        newDataset = pd.DataFrame(columns=allColumns + [DatasetParams.REAL_LEN_COL_NAME])
        protIndex = 0
        featureIndexes = [dataset.columns.get_loc(c) for c in dataset.columns if c in allColumns]
        labelIndex = dataset.columns.get_loc(DatasetParams.LABEL_COL_NAME)
        #print(featureIndexes)
        for i in range(len(protLens)):
            protLen = protLens[i]
            #print(protLen)
            
            # Note: dataset.loc[protIndex:protIndex+protLen, DatasetParams.LABEL_COL_NAME] has a bug. It takes one additional item from the next protein.
            numOnes = np.count_nonzero(dataset.iloc[protIndex:protIndex+protLen, labelIndex])
            if numOnes == 0:
                #print('The protein is removed from dataset due to not having any interface: ', protLen)
                protIndex += protLen
                continue
                
            
            # 'features' is a matrix where columns are part of the original dataset that we have selected, and rows are all residu's of one protein (of length n):
            # 'A'[mean-1 entropy-1 ... length-1]
            # ...
            # 'T'[mean-n entropy-n ... length-n]
            features = dataset.iloc[protIndex:protIndex+protLen, featureIndexes]
            newRow = {}
            for columnInd in range(len(allColumns)):
                columnName = allColumns[columnInd] 
                #print("ColumnName: ", columnName)
                featureInd = features.columns.get_loc(columnName)
                
                # concatenate a specific column for all residu's of this protein (mean-1, ..., mean-n) 
                featureSeq = ",".join(features.iloc[:,featureInd].apply(str))
                #print(featureSeq)
                newRow[allColumns[columnInd]] = featureSeq
            newRow[DatasetParams.REAL_LEN_COL_NAME] = protLen
            if includeId:
                newRow[DatasetParams.PROT_ID_NAME] = protIds[i]
            newDataset = newDataset.append(newRow, ignore_index=True)
            protIndex += protLen 
        
        # Print the list of column names. 
        #print(newDataset.columns.values.tolist())
        #print(newDataset.head(5))
        
        if preparedFileName != None:
            newDataset.to_csv(preparedFileName, index=False) 
        
        return newDataset
    
    @classmethod
    def genVenn3BiolipDS(cls, datasets, labels):
        from collections import Counter
        from matplotlib import pyplot as plt
        from matplotlib_venn import venn3, venn3_circles
        # args = (Abc, aBc, ABc, abC, AbC, aBC, ABC)
        # args = (Psn, pSn, PSn, psN, PsN, pSN, PSN)
        
        dfA = pd.read_csv(datasets[0])
        dfB = pd.read_csv(datasets[1])
        dfC = pd.read_csv(datasets[2])
        A = set(dfA[DatasetParams.PROT_ID_NAME])
        B = set(dfB[DatasetParams.PROT_ID_NAME])
        C = set(dfC[DatasetParams.PROT_ID_NAME])
        AB_overlap = A & B  #compute intersection of set A & set B
        AC_overlap = A & C
        BC_overlap = B & C
        ABC_overlap = A & B & C
        A_rest = A - AB_overlap - AC_overlap #see left graphic
        B_rest = B - AB_overlap - BC_overlap
        C_rest = C - AC_overlap - BC_overlap
        AB_only = AB_overlap - ABC_overlap   #see right graphic
        AC_only = AC_overlap - ABC_overlap
        BC_only = BC_overlap - ABC_overlap
        
        sets = Counter()               #set order A, B, C  
        sets['100'] = len(A_rest)      #100 denotes A on, B off, C off
        sets['001'] = len(C_rest)      #001 denotes A off, B off, C on
        sets['010'] = len(B_rest)      #010 denotes A off, B on, C off
        sets['101'] = len(AC_only)     #101 denotes A on, B off, C on
        sets['110'] = len(AB_only)     #110 denotes A on, B on, C off
        sets['011'] = len(BC_only)     #011 denotes A off, B on, C on
        sets['111'] = len(ABC_overlap) #011 denotes A on, B on, C on
        vennLabels = (labels[0], labels[1], labels[2])
        
        plt.figure(figsize=(7,7)) 
        #plt.title("BioDL_TE Venn diagram")
        #ax = plt.gca() 
        #venn3(subsets=sets, set_labels=labels, ax=ax,set_colors=('darkviolet','deepskyblue','blue'),alpha=0.7)
        venn3(subsets=sets, set_labels=vennLabels, alpha=0.7)    
        plt.show()
        return
    
    @classmethod
    def updateBiolipFile(cls, biolipAFile, biolipPFile, updatedAFile):
        df1 = pd.read_csv(biolipAFile)
        df2 = pd.read_csv(biolipPFile)
        #find all prot-ids that are not common between A and P dataset (prot-ids of N and S datasets)
        nonpids = list(set(df1[DatasetParams.PROT_ID_NAME]).symmetric_difference(df2[DatasetParams.PROT_ID_NAME]))
        print(nonpids)
        #get the indeces of those prot-ids
        nonpinds = df1.index[df1[DatasetParams.PROT_ID_NAME].isin(nonpids)].tolist()
        #print(nonpinds)
        for proti in nonpinds:
            prot_y_true = df1.iloc[proti, df1.columns.get_loc('any_interface')]
            prot_y_true = np.fromstring(prot_y_true, dtype=np.float64, sep=',').tolist()
            prot_y_true = [str(0) for i in prot_y_true]
            prot_y_true = ",".join(prot_y_true)
            #print(prot_y_true)
            df1.loc[proti,'any_interface'] = prot_y_true
        df1.to_csv(updatedAFile, index=False) 
        return
    
    @classmethod
    def prepareBiolipWinData(cls, origFileName, preparedFileName):
        cls.prepareData(origFileName, preparedFileName, labelConv=False, includeId=True)
        
        return
    
    @classmethod
    def prepareCombinedData(cls, homoFile, hetroFile, combinedFile,  includeId=False):    
        print("\n## Generating combined-prepared data for @@: " + combinedFile + " @@")
        homoDataset = cls.prepareData(homoFile, includeId=includeId)
        hetroDataset = cls.prepareData(hetroFile, includeId=includeId)
        
        if os.path.exists(combinedFile):
            os.remove(combinedFile)
        #with open(combinedFile, 'a') as f:    #problem: additional blank rows in the dataset. 
        #    homoDataset.to_csv(f, index=False)
        #   hetroDataset.to_csv(f, index=False, header=False)
        homoDataset.to_csv(combinedFile, index=False)
        hetroDataset.to_csv(combinedFile, mode='a', index=False, header=False)
        
        return
    
    @classmethod    
    def generateProtlenFreq(cls, dataFile, columnName):
        dataSet = pd.read_csv(dataFile, usecols=[columnName])
        
        if DatasetParams.COMET_EXPERIMENT is not None:
            uniqueProtLens = dataSet[columnName].value_counts().index.tolist()
            freqProtLens = dataSet[columnName].value_counts().values.tolist()
            metricName = DatasetPreparation.getStatLabel(dataFile) + 'lenFreq'
            for i in range(0, len(uniqueProtLens)):
                DatasetParams.COMET_EXPERIMENT.log_metric(metricName, freqProtLens[i], step=uniqueProtLens[i])
            return
        else:
            cls.generateHistPlotPL(dataSet, dataFile, columnName)

        return 
    
    @classmethod    
    def generateHistPlotPL(cls, dataSet, dataFile, columnName):
        protLens = dataSet.loc[:, columnName].values
        bins = np.arange(0, dataSet.stack().max() + 1,5)
        datasetName = DatasetPreparation.getStatLabel(dataFile) + DatasetPreparation.getDatasetLabel(dataFile)
        figPlot = go.Figure()
        figPlot.update_layout(
            title=dict(text='Total number of proteins of @@{}@@ : {}'.format(datasetName, dataSet.size)),
            yaxis_title=dict(text='Frequency'),
            xaxis_title=dict(text='Length'),
            showlegend=True,
        )
        figPlot.add_trace(go.Histogram(x=protLens,name='protLenFreq',nbinsx=len(bins)))
        figFile = dataFile.replace('prepared_', 'FREQ_Hist_')
        figFile = figFile.replace('.csv', '.html')
        print("## FREQ-HistPlot generated for: " + figFile)
        plotly.offline.plot(figPlot, filename=figFile, auto_open=False)
        
        return
    
    @classmethod 
    def consFeatureValForAllProts(cls, featureVec):
        def consFeatureValForEachProt(featureStr):
            # Convert '1,0,0,...1,0,1' to [1, 0, ..., 0, 1].
            featureVal = list(eval(featureStr[0]))
            return featureVal
        featureData = np.array(list(map(consFeatureValForEachProt, featureVec)))
        
        return featureData
    
    @classmethod 
    def consAAForAllProts(cls, aavec):
        def consAAForEachProt(aaStr):
            # Convert 'M,D,I,R,...,E,E,A,V' to [M,D,I,R,...,E,E,A,V].
            realAAs = aaStr.split(",")
            return realAAs
        aaData = np.array(list(map(consAAForEachProt, aavec)))
        
        return aaData
    
    @classmethod
    def calcLabelNIPB(cls, protLabel, bins):
        protLen = len(protLabel)
        print('protLen: ', protLen)
        binSize = protLen // bins
        binsLen = bins * binSize
        #print('protLabel: ', protLabel)
        resBins = [protLabel[i:i+binSize] for i in range(0, binsLen, binSize)]
        #print('resBins: ', resBins)
        if protLen > len(resBins):
            # prot has numBins of binSize sublists, and an additional sublist with different size than binSize:
            # [[1,0,3,4],[0,5,0,1],[3,4]]
            # isolate the last bin (which has different size) and remove it from the resBins
            lastBin = protLabel[binsLen:]
            #print('lastBin: ', lastBin)
            # count the number of 1's in the lastBin; 2
            freqLastBin = np.count_nonzero(lastBin)
            # count the number of 0's in the lastBin
            zfreqLastBin = len(lastBin) - freqLastBin
        else:
            # prot has exact numBins of binSize sublists, and no additional sublist with different size than binSize:
            # [[1,0,3,4],[0,5,0,1]]
            freqLastBin = 0
            zfreqLastBin = 0
        # count number of 1's of all sublists (bins); [3,2]   
        freq = np.count_nonzero(resBins, axis=1)
        zfreq = binSize - freq
        #print('freq: ', freq)
        # append number of 1's of the last bin to the resBins; [3,2,2,1] or [3,2,2,0]. Note that the length of freq
        # must always be == numBins + 1
        freq = np.append(freq, freqLastBin)
        zfreq = np.append(zfreq, zfreqLastBin)
        #freq =  (freq / protLen) * bins
        #zfreq =  (zfreq / protLen) * bins
        freq =  (freq / protLen)
        zfreq =  (zfreq / protLen)
        #print('freqLen: ', len(freq), ' freq: ', freq)
        #print('zfreqLen: ', len(zfreq), ' zfreq: ', zfreq)
        
        return freq, zfreq
    
    @classmethod    
    def generateIPBs(cls, dataFile, bins):
        dataSet = pd.read_csv(dataFile)
        labelvec = dataSet.loc[:, DatasetParams.LABEL_COLUMNS].values.astype(np.str)
        labelData = cls.consFeatureValForAllProts(labelvec)
        # (45,) array of list
        #print(labelData.shape)
        #print(labelData)
        freqs = []
        zfreqs = []
        numProts = labelData.size
        numResidues = 0
        numInts = 0
        for l in range(0, numProts):
            protLabel = labelData[l]
            freq, zfreq = cls.calcLabelNIPB(protLabel, bins)
            freqs.append(freq)
            zfreqs.append(zfreq)
            numResidues = numResidues + len(protLabel)
            numInts = numInts +  np.count_nonzero(protLabel)
        # freqs = [[3,2,2,1], [2,4,3,0], ..., [3,0,6,2]]; one element, which is a list of non-zero counts per bin, per protein.
        freqs = np.array(freqs)
        zfreqs = np.array(zfreqs)
        # Shape freqs:  (45, 11)
        #print('Shape freqs: ', freqs.shape)
        
        if DatasetParams.COMET_EXPERIMENT is not None:
            freqsMean = np.mean(freqs, axis=0)
            zfreqsMean = np.mean(zfreqs, axis=0)
            for i in range(0, len(freqsMean)):
                metrics = {
                    DatasetPreparation.getStatLabel(dataFile) + 'IPB': freqsMean[i],
                    DatasetPreparation.getStatLabel(dataFile) + 'NIPB': zfreqsMean[i],
                }
                DatasetParams.COMET_EXPERIMENT.log_metrics(metrics, step=i)
            metrics = {
                DatasetPreparation.getStatLabel(dataFile) + "numProts": numProts,
                DatasetPreparation.getStatLabel(dataFile) + "numResidues": numResidues,
                DatasetPreparation.getStatLabel(dataFile) + "numInts": numInts,
            }
            DatasetParams.COMET_EXPERIMENT.log_metrics(dic=metrics)
        else:
            cls.generateBarPlotIPBs(dataFile, bins, dataSet, freqs, zfreqs)
            cls.generateBoxPlotIPBs(dataFile, bins, dataSet, freqs, zfreqs)
            
        return 
    
    @classmethod    
    def generateBarPlotIPBs(cls, dataFile, bins, dataSet, freqs, zfreqs):
        freqsMean = np.mean(freqs, axis=0)
        zfreqsMean = np.mean(zfreqs, axis=0)
        xs = np.arange(0, bins+1)
        freqSems = stats.sem(freqs, axis=0)
        zfreqSems = stats.sem(zfreqs, axis=0)
        datasetName = DatasetPreparation.getStatLabel(dataFile) + DatasetPreparation.getDatasetLabel(dataFile)
        figPlot = go.Figure()
        figPlot.update_layout(
            title=dict(text='IPB/NIPB bar graph of proteins of @@{}@@'.format(datasetName)),
            yaxis_title=dict(text='Mean IPBs/NIPBs (normalized # of I/NI per bin)'),
            xaxis_title=dict(text='Bins (proteins divided in bins)'),
            showlegend=True,
            barmode="overlay",
        )
        figPlot.add_trace(go.Bar(y=freqsMean,x=xs,name='IPB',error_y=dict(type='data',array=freqSems*2),opacity=0.7))
        figPlot.add_trace(go.Bar(y=zfreqsMean,x=xs,name='NIPB',error_y=dict(type='data',array=zfreqSems*2),opacity=0.4))
        figFile = dataFile.replace('prepared_', 'IPB_Bar_')
        figFile = figFile.replace('.csv', '.html')
        print("## IPB-BarPlot generated for: " + figFile)
        plotly.offline.plot(figPlot, filename=figFile, auto_open=False)
        
        return
    
    @classmethod    
    def generateBoxPlotIPBs(cls, dataFile, bins, dataSet, freqs, zfreqs):
        xs = np.arange(0, bins+1)
        tfreqs = np.transpose(freqs)
        protIds = dataSet.loc[:, DatasetParams.PROT_ID_NAME].values
        datasetName = DatasetPreparation.getStatLabel(dataFile) + DatasetPreparation.getDatasetLabel(dataFile)
        figPlot = go.Figure()
        figPlot.update_layout(
            title=dict(text='IPB boxplot of proteins of @@{}@@'.format(datasetName)),
            yaxis_title=dict(text='IPBs (normalized # of interfaces per bin)'),
            xaxis_title=dict(text='Bins (proteins divided in bins)'),
            hoverlabel = dict(bgcolor='lightgreen'),
        )
        for i in range(0, len(xs)):
            text = '<a href=\"https://www.uniprot.org/uniprot/' + protIds + '\">' + protIds + '</a>'
            figPlot.add_trace(go.Box(y=tfreqs[i],name=str(xs[i]),boxpoints='all',hovertext=text))
        figFile = dataFile.replace('prepared_', 'IPB_Box_')
        figFile = figFile.replace('.csv', '.html')
        print("## IPB-BoxPlot generated for: " + figFile)
        plotly.offline.plot(figPlot, filename=figFile, auto_open=False)
        
        return
    
    @classmethod
    def calcAANIPB(cls, protLabel, protAA, bins):
        def groupProtAAs(labelResBins, aaResBins, calcZFreq):
            # labelResBins =  [[2,3,4,5],[8,10,12],[]] 
            # aaResBins = [['a','b','c','a'], ['a','f','f'],[]]
            # result = normalized #IAA/NIAA of this protein per bin (3): [{'a': 3.5, 'b': 1.5, 'c': 2.0}, {'a': 4.0, 'f': 11.0}, {}]

            def groupAAsForEachBin(labelResBin, aaResBin):
                # labelResBin:  [2,3,4,5]
                # aaResBin: ['a','b','c','a']
                # result = normalized #IAA/NIAA of this protein for one bin: {'a': 3.5, 'b': 1.5, 'c': 2.0}; 
                # it can be an empty dict {} if lastBin is empty.
                
                def doHist(g):
                    realVals = g[0].values
                    freq = np.count_nonzero(realVals)
                    if calcZFreq:
                        freq = len(realVals) - freq
                    return freq

                dfBin = pd.DataFrame(labelResBin, index=aaResBin)
                dfBinFreq = dfBin.groupby(aaResBin).apply(doHist)
                # normalize to protlen
                dfBinFreq.loc[dfBinFreq.index] = dfBinFreq.values / protLen
                if dfBinFreq.empty: 
                    dictBin = dict()
                else:
                    dictBin = dfBinFreq.to_dict()
                
                return dictBin
            
            #print('aaResBins[0]: ', aaResBins[0])
            #print('labelResBins[0]: ', labelResBins[0])
            dictBins = list(map(groupAAsForEachBin, labelResBins, aaResBins))
            return dictBins

        
        protLen = len(protLabel)
        #print('protLen: ', protLen)
        binWidth = protLen // bins
        binsLen = bins * binWidth
        #print('protLabel: ', protLabel)
        labelResBins = [protLabel[i:i+binWidth] for i in range(0, binsLen, binWidth)]
        aaResBins = [protAA[i:i+binWidth] for i in range(0, binsLen, binWidth)]
        #print('labelResBins: ', labelResBins)
        if protLen > len(labelResBins):
            # prot has numBins of binWidth sublists, and an additional sublist with different size than binWidth:
            # [[1,0,3,4],[0,5,0,1],[3,4]]
            # isolate the last bin (which has different size) and remove it from the resBins
            labelLastBin = protLabel[binsLen:]
            aaLastBin = protAA[binsLen:]
        else:
            labelLastBin = []
            aaLastBin = []
        labelResBins.append(labelLastBin)
        aaResBins.append(aaLastBin)
        #print('ffffffffffffff freqs ==============')
        freq = groupProtAAs(labelResBins, aaResBins, False)
        #print('zzzzzzzzzzzzzz zfreqs ==============')
        zfreq = groupProtAAs(labelResBins, aaResBins, True)
        
        #freq,zfreq = normalized #IAA/NIAA of this protein per bin (3): [{'a': 3.5, 'b': 1.5, 'c': 2.0}, {'a': 4.0, 'f': 11.0}, {}]
        return freq, zfreq
    
    @classmethod    
    def generateAAIPBs(cls, dataFile, bins, genPlot=True):
        def gatherAAsForEach(aaDict1, aaDict2):
            # aaDict1={'a': [], 'h': [], 's': [], 'f': [], 'c': [], 'b': []}
            # aaDict2={'a': 0.5, 'b': 0.5, 'c': 0.5}
            def appendToList(k):
                #print('k: ', k)
                aaDict1List = aaDict1.get(k).copy()
                aaDict1List.append(aaDict2.get(k,0.0))
                return aaDict1List
        
            gatheredAADict = {k: appendToList(k) for k in aaDict1.keys() or aaDict2.keys()}
            # gatheredAADict={'a': [0.5], 'h': [0.0], 's': [0.0], 'f': [0.0], 'c': [0.5], 'b': [0.5]}
            return gatheredAADict
        
        def getFreqsMeanAndSumAndStd(freqsPerProtPerBinPerAADict):
            # parameter freqsPerProtPerBinPerAADict = [ prot-1=[b1,b2,b3], prot-2=[b1,b2,b3], prot-3=[b1,b2,b3] ] and bx={AA: aggre-value-within-one-prot}
            # [ [{'a': 0.5, 'b': 0.5, 'c': 0.5}, {'a': 0.5, 'f': 1.0, 's': 0.0}, {}],
            #   [{'a': 1.0, 'c': 0.5, 'h': 0.5}, {'a': 0.5, 'f': 0.5, 's': 0.5}, {}],
            #   [{'a': 1.0, 'b': 0.5, 'c': 0.5}, {'a': 0.5, 'f': 1.0}, {'h': 0.5, 's': 0.5}],
            #   [{'a': 0.5, 'b': 0.5, 's': 1.0}, {'a': 0.5, 'f': 0.5, 'h': 0.5}, {'a': 0.5}]
            # ]
            
            # transpose from protein-level to bin-level.
            freqsPerBinPerAADictPerProt = np.array(freqsPerProtPerBinPerAADict).T.tolist()
            # freqsPerBinPerAADictPerProt = [ bin-1=[p1b1,p2b1,p3b1], bin-2=[p1b2,p2b2,p3b2], bin-3=[p1b3,p2b3,p3b3] ]
            # [[{'a': 0.5, 'b': 0.5, 'c': 0.5}, {'a': 1.0, 'c': 0.5, 'h': 0.5}, {'a': 1.0, 'b': 0.5, 'c': 0.5}, {'a': 0.5, 'b': 0.5, 's': 1.0}],
            #  [{'a': 0.5, 'f': 1.0, 's': 0.0}, {'a': 0.5, 'f': 0.5, 's': 0.5}, {'a': 0.5, 'f': 1.0}, {'a': 0.5, 'f': 0.5, 'h': 0.5}], 
            #  [{}, {}, {'h': 0.5, 's': 0.5}, {'a': 0.5}]
            # ]
            
            dictPerBinPerAAPerProt = list(map(lambda b: functools.reduce(gatherAAsForEach,b,initAAs), freqsPerBinPerAADictPerProt))
            # gatheredFreqs = [ bin-1={aa:[], ..., bb=[]}, ..., bin-n={aa:[], ..., bb=[]} ]
            # [{'a': [0.5, 1.0, 1.0, 0.5], 'h': [0.0, 0.5, 0.0, 0.0], 's': [0.0, 0.0, 0.0, 1.0], 'f': [0.0, 0.0, 0.0, 0.0], 'c': [0.5, 0.5, 0.5, 0.0], 'b': [0.5, 0.0, 0.5, 0.5]}, 
            #  {'a': [0.5, 0.5, 0.5, 0.5], 'h': [0.0, 0.0, 0.0, 0.5], 's': [0.0, 0.5, 0.0, 0.0], 'f': [1.0, 0.5, 1.0, 0.5], 'c': [0.0, 0.0, 0.0, 0.0], 'b': [0.0, 0.0, 0.0, 0.0]}, 
            #  {'a': [0.0, 0.0, 0.0, 0.5], 'h': [0.0, 0.0, 0.5, 0.0], 's': [0.0, 0.0, 0.5, 0.0], 'f': [0.0, 0.0, 0.0, 0.0], 'c': [0.0, 0.0, 0.0, 0.0], 'b': [0.0, 0.0, 0.0, 0.0]}
            # ]

            valPerBinPerAAPerProt = list(map(lambda aaDict: [v for v in aaDict.values()], dictPerBinPerAAPerProt))
            # valPerBinPerAAPerProt = count-per-dim1(bin)-per-dim2(AA)-per-dim-3(prot)
            # [[[0.5, 1.0, 1.0, 0.5], [0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.5, 0.0], [0.5, 0.0, 0.5, 0.5]], 
            #  [[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.5], [0.0, 0.5, 0.0, 0.0], [1.0, 0.5, 1.0, 0.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            #  [[0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            # ]
            
            meanPerAAPerBin = np.mean(valPerBinPerAAPerProt, axis=2).T
            # meanPerAAPerBin:  
            # [[0.75, 0.5, 0.125], [0.125, 0.125, 0.125], [0.25, 0.125, 0.125], [0.0, 0.75, 0.0], [0.375, 0.0, 0.0], [0.375, 0.0, 0.0]]
            
            stdPerAAPerBin = np.std(valPerBinPerAAPerProt, axis=2).T
            # stdPerAAPerBin: 
            # [[0.25, 0.0, 0.21650635094610965], [0.21650635094610965, 0.21650635094610965, 0.21650635094610965],
            #  [0.4330127018922193, 0.21650635094610965, 0.21650635094610965], [0.0, 0.25, 0.0], [0.21650635094610965, 0.0, 0.0],
            #  [0.21650635094610965, 0.0, 0.0]]
            
            sumPerAAPerProt = np.sum(valPerBinPerAAPerProt, axis=0)
            # sumPerAAPerProt:  
            # [[1.0, 1.5, 1.5, 1.5], [0.0, 0.5, 0.5, 0.5], [0.0, 0.5, 0.5, 1.0], [1.0, 0.5, 1.0, 0.5], [0.5, 0.5, 0.5, 0.0], [0.5, 0.0, 0.5, 0.5]]

            globMeanPerAA = np.mean(sumPerAAPerProt, axis=1)
            # globMeanPerAA:  [1.375, 0.375, 0.5, 0.75, 0.375, 0.375]
            
            globStdPerAA = np.std(sumPerAAPerProt, axis=1)
            # globStdPerAA:  [0.21650635094610965, 0.21650635094610965, 0.3535533905932738, 0.25, 0.21650635094610965, 0.21650635094610965]
            
            return meanPerAAPerBin, stdPerAAPerBin, globMeanPerAA, globStdPerAA
        
        def calcAAPercents():    
            totalMeanPerAAPerBin = meanPerAAPerBin + zmeanPerAAPerBin
            binAAPercent = (np.round( (meanPerAAPerBin * 100) / totalMeanPerAAPerBin, 1)) / 100
            #binAAPercent = np.nan_to_num(binAAPercent, copy=False)
            totalGlobMeanPerAA = globMeanPerAA + zglobMeanPerAA
            globAAPercent = (np.round( (globMeanPerAA * 100) / totalGlobMeanPerAA, 1)) / 100
            
            return binAAPercent, globAAPercent
        
        dataSet = pd.read_csv(dataFile)
        labelvec = dataSet.loc[:, DatasetParams.LABEL_COLUMNS].values.astype(np.str)
        labelData = cls.consFeatureValForAllProts(labelvec)
        aavec = dataSet.loc[:, DatasetParams.SEQ_COLUMN_NAME].values.astype(np.str)
        aaData = cls.consAAForAllProts(aavec)
        # (45,) array of list
        #print(aaData.shape)
        #print(aaData)
        freqs = []
        zfreqs = []
        numProts = labelData.size
        numResidues = 0
        numInts = 0
        for l in range(0, numProts):
            protLabel = labelData[l]
            protAA = aaData[l]
            freq, zfreq = cls.calcAANIPB(protLabel, protAA, bins)
            freqs.append(freq)
            zfreqs.append(zfreq)
            numResidues = numResidues + len(protLabel)
            numInts = numInts +  np.count_nonzero(protLabel)
        
        percenInts = round((numInts / numResidues) * 100, 1)
        statData = [numProts, numInts, percenInts]
        
        initAAs = dict.fromkeys(DatasetParams.AA_LIST, [])
        meanPerAAPerBin,stdPerAAPerBin,globMeanPerAA,globStdPerAA = getFreqsMeanAndSumAndStd(freqs)
        zmeanPerAAPerBin,_,zglobMeanPerAA,_ = getFreqsMeanAndSumAndStd(zfreqs)
        binAAPercent,globAAPercent = calcAAPercents()
        
        if genPlot:
            cls.generateBarPlotAAs(dataFile, globMeanPerAA, zglobMeanPerAA, statData)
            cls.generateBarPlotAAIPBs(dataFile, bins, meanPerAAPerBin, zmeanPerAAPerBin, ['AAIPB','IAA'], statData)
            cls.generateBarPlotAAIPBs(dataFile, bins, zmeanPerAAPerBin, meanPerAAPerBin, ['AANIPB','NIAA'], statData)
        
        # note: all these variables are a numpy array (and not a list)
        return globMeanPerAA,globStdPerAA,globAAPercent,meanPerAAPerBin,stdPerAAPerBin,binAAPercent
    
    @classmethod    
    def generateStatFile(cls, dataFile, bins):
        statFile = dataFile.replace('prepared_', 'Stat_')
        globMeanPerAA,globStdPerAA,globAAPercent,meanPerAAPerBin,stdPerAAPerBin,binAAPercent = cls.generateAAIPBs(dataFile, bins, False)
        statCols = ['glob_mean', 'glob_std', 'glob_percent']
        statData = (globMeanPerAA,globStdPerAA,globAAPercent)
        statData = np.vstack(statData).T
        dataset = pd.DataFrame(statData, columns=statCols, index=DatasetParams.AA_LIST)
        print("\n## Generating stat-data for @@: " + statFile + " @@")
        dataset.to_csv(statFile, index=True) 
        return
    
    @classmethod    
    def addStatFeatures(cls, dataFile, bins):
        def consPPSMAAList():
            pssmAAList = list(map(lambda aa: 'pssm_'+aa, DatasetParams.AA_LIST))
            return pssmAAList

        def calcAAFreq(aaResBins, protLen):
            # aaResBins = [['a','b','c','a','a'], ['a','f','f','s'],[]]
            def calcAAFreqForEachBin(aaResBin):
                # aaResBin = ['a','b','c','a','a']
                dfBin = pd.DataFrame(aaResBin)
                dfBinFreq = dfBin.groupby(aaResBin).count()
                # normalize to protlen
                dfBinFreq.loc[dfBinFreq.index] = dfBinFreq.values / protLen
                if dfBinFreq.empty: 
                    dictBinFreq = dict()
                else:
                    dictBinFreq = dfBinFreq.to_dict()[0]
                freqPerAAForBin = {k: dictBinFreq.get(k,0.0) for k in initAAs.keys()}
                # freqPerAAForBin:  {'a': 3, 'h': 0.0, 's': 0.0, 'f': 0.0, 'c': 1, 'b': 1}
                return list(freqPerAAForBin.values())   # freqPerAAForBin: [3, 0.0, 0.0, 0.0, 1, 1]
            
            initAAs = dict.fromkeys(DatasetParams.AA_LIST, 0.0)
            freqPerBinPerAA = list(map(calcAAFreqForEachBin, aaResBins))
            # freqPerBinPerAA:  [[3, 0.0, 0.0, 0.0, 1, 1], [1, 0.0, 1, 2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
            globFreqPerAA = np.sum(freqPerBinPerAA, axis=0)
            # globFreqPerAA:  [4. 0. 1. 2. 1. 1.]
            
            return globFreqPerAA,freqPerBinPerAA
        
        def ConsAAFreqForProt(protAA, bins):
            protLen = len(protAA)
            binWidth = protLen // bins
            binsLen = bins * binWidth
            aaResBins = [protAA[i:i+binWidth] for i in range(0, binsLen, binWidth)]
            if protLen > len(aaResBins):
                aaLastBin = protAA[binsLen:]
            else:
                aaLastBin = []
            aaResBins.append(aaLastBin)
            
            return calcAAFreq(aaResBins, protLen)   # normalized globFreqPerAA and freqPerBinPerAA  
                
        def consPSSMsForProt(proti, protLen, pssmAAList): 
            def consPSSMValForEachPSSM(pssmLabel):
                try:
                    pssmStr = protDataset.loc[proti, pssmLabel]
                    pssmVal = list(eval(pssmStr))
                except:
                    pssmVal = [0.0 for aa in range(protLen)]
                return pssmVal
            
            pssmData = np.array(list(map(consPSSMValForEachPSSM, pssmAAList)))
            # pssmData = [ [pssm_A-values-perAA-position], ..., [pssm_X-values which are all zero's perAA-position] ]
            # [ [0.04743 0.8808  0.73106 ... 0.8808  0.95257 0.99331]
            #   [0.      0.      0.      ... 0.      0.      0.     ] ]
            return pssmData
        
        #def addStatCols(protDataset):
        #    protDataset['glob_stat_score'] = ''
        #    for b in range(0, bins+1):
        #        protDataset['bin'+str(b)+'_stat_score'] = ''
        #    return protDataset
        
        def calcStatScoreForProt(proti, protAA, globFreqPerAA, freqPerBinPerAA, pssmData):
            def caIcIntZscoreAA(intMean, intStd, intPercent, aaFreq):
                # these are calculated once per AA-type for a specific protein. In this way, we can avoid calculation for 
                # an AA-type in multiple positions.
                intFreq = intPercent * aaFreq
                intZscoreAA = (intFreq - intMean) / intStd
                #intZscoreAA = intZscoreAA / intFreq  #zscore per-normalized-int-freq for an AA
                #intZscoreAA = stats.norm.cdf(intZscoreAA, intMean, intStd)
                intZscoreAA = 1 / (1 + math.exp(-intZscoreAA))
                #intZscoreAA = intZscoreAA * intPercent
                return intZscoreAA
            
            def calcStatScore(zscoreData, pssmData):
                statScoreList = []
                for pos in range(len(protAA)):
                    aa = protAA[pos]
                    aai = DatasetParams.AA_LIST.index(aa)
                    zscoreAA = zscoreData[aai]
                    pssmAAPerPos = pssmData[aai,pos]
                    #pssmAAPerPos = 1
                    statScoreAAPerPos = round(zscoreAA * pssmAAPerPos, 5)
                    #statScoreAAPerPos = 1 / (1 + math.exp(-statScoreAAPerPos))
                    statScoreList.append(statScoreAAPerPos)
                statScoreSeq = ",".join(str(e) for e in statScoreList)
                return statScoreSeq
            
            def addStatCol(statCol, statVal):
                protDataset.loc[proti,statCol] = statVal
                return
            
            statScoreDict = {}
            globZscoreData = []
            for i in range(len(DatasetParams.AA_LIST)):
                globMean = statDataset.loc[i,'glob_mean']
                globStd = statDataset.loc[i,'glob_std']
                globPercent = statDataset.loc[i,'glob_percent']
                aaGlobFreq = globFreqPerAA[i]
                globZscoreAA = caIcIntZscoreAA(globMean, globStd, globPercent, aaGlobFreq)
                globZscoreData.append(globZscoreAA)
                # do the same for bins
            globStatVal = calcStatScore(globZscoreData, pssmData)
            addStatCol('glob_stat_score', globStatVal)
            return statScoreDict
        
        def saveProtDataset():
            tmpFile = dataFile.replace('data/', 'data/stat-files/')
            protDataset.to_csv(tmpFile, index=False)
            print("\n## Generating new prot-data for @@: " + tmpFile + " @@")
            return
        
        protDataset = pd.read_csv(dataFile)
        statFile = cls.getStatFileName(dataFile)
        statDataset = pd.read_csv(statFile)
        
        pssmAAList = consPPSMAAList()
        aavec = protDataset.loc[:, DatasetParams.SEQ_COLUMN_NAME].values.astype(np.str)
        aaData = cls.consAAForAllProts(aavec)
        numProts = aaData.size
        for proti in range(0, numProts):
            protAA = aaData[proti]
            globFreqPerAA,freqPerBinPerAA = ConsAAFreqForProt(protAA, bins)
            protLen = len(protAA)
            pssmData = consPSSMsForProt(proti, protLen, pssmAAList)
            calcStatScoreForProt(proti, protAA, globFreqPerAA, freqPerBinPerAA, pssmData)
        saveProtDataset()
        return
    
    @classmethod    
    def generateBarPlotAAs(cls, dataFile, globMeanPerAA, zglobMeanPerAA, statData):
        def getHoverTextList(isNI):
            textList = []
            for i in range(0, len(xs)):
                aa1LetCode = xs[i]
                totalSum = globMeanPerAA[i] + zglobMeanPerAA[i]
                if (isNI):
                    aaPercent = np.round( (zglobMeanPerAA[i] * 100) / totalSum, 1)
                else:
                    aaPercent = np.round( (globMeanPerAA[i] * 100) / totalSum, 1)
                text = '<a href=\"http://www.bmrb.wisc.edu/referenc/commonaa.php?' + \
                    cls.getAALabel(aa1LetCode)[1] + '\">' + cls.getAALabel(aa1LetCode)[0] + ' (' + str(aaPercent) + '%)</a>'
                textList.append(text)
            return textList
               
        xs = DatasetParams.AA_LIST
        datasetName = DatasetPreparation.getStatLabel(dataFile) + DatasetPreparation.getDatasetLabel(dataFile)
        figPlot = go.Figure()
        figPlot.update_layout(
            title=dict(text='IAA/NIAA bar graph of proteins of @@{}@@<br>[#Proteins={}, #Interfaces={}, Interfaces={}%]'.format(datasetName,statData[0],statData[1],statData[2])),
            yaxis_title=dict(text='Mean IAAs/NIAAs (normalized # of I/NI per AA)'),
            xaxis_title=dict(text='Amino Acids occurring in all proteins'),
            showlegend=True,
            barmode="overlay",
            hoverlabel=dict(bgcolor='lightgreen'),
            #hovermode='closest',
            #clickmode='event+select',
        )
        figPlot.add_trace(go.Bar(y=globMeanPerAA,x=xs,name='IAA',hovertext=getHoverTextList(False),opacity=0.7))
        figPlot.add_trace(go.Bar(y=zglobMeanPerAA,x=xs,name='NIAA',hovertext=getHoverTextList(True),opacity=0.4))
        figFile = dataFile.replace('prepared_', 'IAA_Bar_')
        figFile = figFile.replace('.csv', '.html')
        print("## IAA-BarPlot generated for: " + figFile)
        plotly.offline.plot(figPlot, filename=figFile, auto_open=False)
        
        return
    
    @classmethod    
    def generateBarPlotAAIPBs(cls, dataFile, bins, meanPerAAPerBin, cmeanPerAAPerBin, aaTitles, statData):
        # [cmeanPerAAPerBin = zfreqsMean] if we want to plot interfaces; otherwise it's value is 'meanPerAAPerBin' and [meanPerAAPerBin = zfreqsMean].
        xs = np.arange(0, bins+1)
        datasetName = DatasetPreparation.getStatLabel(dataFile) + DatasetPreparation.getDatasetLabel(dataFile)
        figPlot = go.Figure()
        figPlot.update_layout(
            title=dict(text=aaTitles[0]+' bar graph of proteins of @@{}@@<br>[#Proteins={}, #Interfaces={}, Interfaces={}%]'.format(datasetName,statData[0],statData[1],statData[2])),
            yaxis_title=dict(text='Mean '+aaTitles[0]+'s (normalized # of '+aaTitles[1]+' per bin)'),
            xaxis_title=dict(text='Bins (proteins divided in bins)'),
            showlegend=True,
            barmode="stack",
            hoverlabel = dict(bgcolor='lightgreen'),
        )
        for i in range(0, len(meanPerAAPerBin)):
            aa1LetCode = DatasetParams.AA_LIST[i]
            totalMean = meanPerAAPerBin[i] + cmeanPerAAPerBin[i]
            aaPercent = np.round( (meanPerAAPerBin[i] * 100) / totalMean, 1)
            textList = []
            for b in range(0, xs.size):
                text = '<a href=\"http://www.bmrb.wisc.edu/referenc/commonaa.php?' + \
                    cls.getAALabel(aa1LetCode)[1] + '\">' + cls.getAALabel(aa1LetCode)[0] + ' (' + str(aaPercent[b]) + '%)</a>'
                textList.append(text)
            figPlot.add_trace(go.Bar(y=meanPerAAPerBin[i],x=xs,name=aa1LetCode,hovertext=textList))
                              
        figFile = dataFile.replace('prepared_', aaTitles[0]+'_Bar_')
        figFile = figFile.replace('.csv', '.html')
        print('## '+aaTitles[0]+'-BarPlot generated for: ' + figFile)
        #plotly.offline.plot(figPlot, filename=figFile, auto_open=False, include_plotlyjs='cdn')
        plotly.offline.plot(figPlot, filename=figFile, auto_open=False)
        
        return
    
def genPreparedSerendipFiles(onlyTest=False):
    DatasetParams.ALL_COLUMNS = DatasetParams.FEATURE_COLUMNS_ENH_WIN
    DatasetParams.LABEL_COL_NAME = 'Interface1'
    DatasetPreparation.prepareData(DatasetParams.HOMO_TESTING_FILE, DatasetParams.PREPARED_HOMO_TESTING_FILE, includeId=True)
    DatasetPreparation.prepareData(DatasetParams.HETRO_TESTING_FILE, DatasetParams.PREPARED_HETRO_TESTING_FILE, includeId=True)
    """
    DatasetPreparation.prepareData(DatasetParams.HOMO_TRAINING_FILE, DatasetParams.PREPARED_HOMO_TRAINING_FILE, includeId=True)
    DatasetPreparation.prepareData(DatasetParams.HOMO_TESTING_FILE, DatasetParams.PREPARED_HOMO_TESTING_FILE, includeId=True)
    DatasetPreparation.prepareData(DatasetParams.HETRO_TRAINING_FILE, DatasetParams.PREPARED_HETRO_TRAINING_FILE, includeId=True)
    DatasetPreparation.prepareData(DatasetParams.HETRO_TESTING_FILE, DatasetParams.PREPARED_HETRO_TESTING_FILE, includeId=True)
    DatasetPreparation.prepareCombinedData(DatasetParams.HOMO_TRAINING_FILE, DatasetParams.HETRO_TRAINING_FILE, DatasetParams.PREPARED_COMBINED_TRAINING_FILE, includeId=True)
    DatasetPreparation.prepareCombinedData(DatasetParams.HOMO_TESTING_FILE, DatasetParams.HETRO_TESTING_FILE, DatasetParams.PREPARED_COMBINED_TESTING_FILE, includeId=True)
    """
    return

def genPreparedEpitopeFiles(onlyTest=False):
    DatasetParams.ALL_COLUMNS = DatasetParams.FEATURE_COLUMNS_EPI_WIN
    DatasetParams.LABEL_COL_NAME = 'Interface1'
    if not onlyTest:
        DatasetPreparation.prepareData(DatasetParams.EPITOPE_TRAINING_FILE, DatasetParams.PREPARED_EPITOPE_TRAINING_FILE, includeId=True)
    DatasetPreparation.prepareData(DatasetParams.EPITOPE_TESTING_FILE, DatasetParams.PREPARED_EPITOPE_TESTING_FILE, includeId=True)
    return

def genPreparedBiolipFiles(onlyTest=False):
    DatasetParams.ALL_COLUMNS = DatasetParams.FEATURE_COLUMNS_BIOLIP_WIN
    #"""
    DatasetParams.LABEL_COL_NAME = 'p_interface'
    DatasetPreparation.prepareBiolipWinData(DatasetParams.BIOLIP_WIN_TRAINING_FILE, DatasetParams.PREPARED_BIOLIP_WIN_P_TRAINING_FILE)
    DatasetPreparation.prepareBiolipWinData(DatasetParams.BIOLIP_WIN_TESTING_FILE, DatasetParams.PREPARED_BIOLIP_WIN_P_TESTING_FILE)
    DatasetPreparation.prepareBiolipWinData(DatasetParams.ZK448_WIN_BENCHMARK_FILE, DatasetParams.PREPARED_ZK448_WIN_P_BENCHMARK_FILE)
    DatasetParams.LABEL_COL_NAME = 'n_interface'
    DatasetPreparation.prepareBiolipWinData(DatasetParams.BIOLIP_WIN_TRAINING_FILE, DatasetParams.PREPARED_BIOLIP_WIN_N_TRAINING_FILE)
    DatasetPreparation.prepareBiolipWinData(DatasetParams.BIOLIP_WIN_TESTING_FILE, DatasetParams.PREPARED_BIOLIP_WIN_N_TESTING_FILE)
    DatasetPreparation.prepareBiolipWinData(DatasetParams.ZK448_WIN_BENCHMARK_FILE, DatasetParams.PREPARED_ZK448_WIN_N_BENCHMARK_FILE)
    DatasetParams.LABEL_COL_NAME = 's_interface'
    DatasetPreparation.prepareBiolipWinData(DatasetParams.BIOLIP_WIN_TRAINING_FILE, DatasetParams.PREPARED_BIOLIP_WIN_S_TRAINING_FILE)
    DatasetPreparation.prepareBiolipWinData(DatasetParams.BIOLIP_WIN_TESTING_FILE, DatasetParams.PREPARED_BIOLIP_WIN_S_TESTING_FILE)
    DatasetPreparation.prepareBiolipWinData(DatasetParams.ZK448_WIN_BENCHMARK_FILE, DatasetParams.PREPARED_ZK448_WIN_S_BENCHMARK_FILE)
    DatasetParams.LABEL_COL_NAME = 'any_interface'
    DatasetPreparation.prepareBiolipWinData(DatasetParams.BIOLIP_WIN_TRAINING_FILE, DatasetParams.PREPARED_BIOLIP_WIN_A_TRAINING_FILE)
    DatasetPreparation.prepareBiolipWinData(DatasetParams.BIOLIP_WIN_TESTING_FILE, DatasetParams.PREPARED_BIOLIP_WIN_A_TESTING_FILE)
    DatasetPreparation.prepareBiolipWinData(DatasetParams.ZK448_WIN_BENCHMARK_FILE, DatasetParams.PREPARED_ZK448_WIN_A_BENCHMARK_FILE)
    #"""
    return

def genSerendipStats(use_comet=False):
    DatasetParams.SEQ_COLUMN_NAME = 'AliSeq'
    DatasetParams.LABEL_COLUMNS = ['Interface1']
    projName = 'serendip-ds'
    if use_comet:
        DatasetPreparation.initExperiment(projName, 'Combined')    
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_COMBINED_TRAINING_FILE, DatasetParams.REAL_LEN_COL_NAME) 
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_COMBINED_TRAINING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_COMBINED_TRAINING_FILE, 10)
    
    if use_comet:
        DatasetPreparation.initExperiment(projName, 'Hetro')
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_HETRO_TESTING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_HETRO_TESTING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_HETRO_TESTING_FILE, 10)
    
    if use_comet:
        DatasetPreparation.initExperiment(projName, 'Homo')
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_HOMO_TESTING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_HOMO_TESTING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_HOMO_TESTING_FILE, 10)
    return

def genEpitopeStats(use_comet=False):
    DatasetParams.SEQ_COLUMN_NAME = 'AliSeq'
    DatasetParams.LABEL_COLUMNS = ['Interface1']
    projName = 'epitope-ds'
    
    if use_comet:
        DatasetPreparation.initExperiment(projName, 'Epitope_1')
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_EPITOPE_1_TRAINING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_EPITOPE_1_TESTING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_EPITOPE_1_TRAINING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_EPITOPE_1_TESTING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_EPITOPE_1_TRAINING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_EPITOPE_1_TESTING_FILE, 10)    
    
    if use_comet:
        DatasetPreparation.initExperiment(projName, 'Epitope_2')
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_EPITOPE_2_TRAINING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_EPITOPE_2_TESTING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_EPITOPE_2_TRAINING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_EPITOPE_2_TESTING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_EPITOPE_2_TRAINING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_EPITOPE_2_TESTING_FILE, 10)    
    
    if use_comet:
        DatasetPreparation.initExperiment(projName, 'Epitope_3')
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_EPITOPE_3_TRAINING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_EPITOPE_3_TESTING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_EPITOPE_3_TRAINING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_EPITOPE_3_TESTING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_EPITOPE_3_TRAINING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_EPITOPE_3_TESTING_FILE, 10)    
    
    if use_comet:
        DatasetPreparation.initExperiment(projName, 'Epitope_4')
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_EPITOPE_4_TRAINING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_EPITOPE_4_TESTING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_EPITOPE_4_TRAINING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_EPITOPE_4_TESTING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_EPITOPE_4_TRAINING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_EPITOPE_4_TESTING_FILE, 10)    
    
    if use_comet:
        DatasetPreparation.initExperiment(projName, 'Epitope_5')
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_EPITOPE_5_TRAINING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_EPITOPE_5_TESTING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_EPITOPE_5_TRAINING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_EPITOPE_5_TESTING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_EPITOPE_5_TRAINING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_EPITOPE_5_TESTING_FILE, 10)    
    return

def addBiolipStatFeatures():
    DatasetParams.SEQ_COLUMN_NAME = 'sequence'

    DatasetParams.LABEL_COLUMNS = ['n_interface']
    DatasetPreparation.addStatFeatures(DatasetParams.PREPARED_BIOLIP_WIN_N_TRAINING_FILE, 3)
    DatasetPreparation.addStatFeatures(DatasetParams.PREPARED_BIOLIP_WIN_N_TESTING_FILE, 3)
    DatasetPreparation.addStatFeatures(DatasetParams.PREPARED_ZK448_WIN_N_BENCHMARK_FILE, 3) #10,30,60,3
    return
    
def genBiolipStatFile():
    DatasetParams.SEQ_COLUMN_NAME = 'sequence'

    DatasetParams.LABEL_COLUMNS = ['n_interface']
    #DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_ZK448_WIN_N_BENCHMARK_FILE, 3, True) #10,30,60,3
    DatasetPreparation.generateStatFile(DatasetParams.PREPARED_BIOLIP_WIN_N_TRAINING_FILE, 3) #10,30,60,
    return
        
def genBiolipStats(use_comet=False):
    DatasetParams.SEQ_COLUMN_NAME = 'sequence'
    projName = 'biolip-ds'

    if use_comet:
        DatasetPreparation.initExperiment(projName, 'Biolip_N')
    DatasetParams.LABEL_COLUMNS = ['n_interface']
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_BIOLIP_WIN_N_TRAINING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_BIOLIP_WIN_N_TESTING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_ZK448_WIN_N_BENCHMARK_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_BIOLIP_WIN_N_TRAINING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_BIOLIP_WIN_N_TESTING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_ZK448_WIN_N_BENCHMARK_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_BIOLIP_WIN_N_TRAINING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_BIOLIP_WIN_N_TESTING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_ZK448_WIN_N_BENCHMARK_FILE, 10)
   
    if use_comet:
        DatasetPreparation.initExperiment(projName, 'Biolip_P')
    DatasetParams.LABEL_COLUMNS = ['p_interface']
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_BIOLIP_WIN_P_TRAINING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_BIOLIP_WIN_P_TESTING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_ZK448_WIN_P_BENCHMARK_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_BIOLIP_WIN_P_TRAINING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_BIOLIP_WIN_P_TESTING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_ZK448_WIN_P_BENCHMARK_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_BIOLIP_WIN_P_TRAINING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_BIOLIP_WIN_P_TESTING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_ZK448_WIN_P_BENCHMARK_FILE, 10)
    
    if use_comet:
        DatasetPreparation.initExperiment(projName, 'Biolip_S')
    DatasetParams.LABEL_COLUMNS = ['s_interface']
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_BIOLIP_WIN_S_TRAINING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_BIOLIP_WIN_S_TESTING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_ZK448_WIN_S_BENCHMARK_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_BIOLIP_WIN_S_TRAINING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_BIOLIP_WIN_S_TESTING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_ZK448_WIN_S_BENCHMARK_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_BIOLIP_WIN_S_TRAINING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_BIOLIP_WIN_S_TESTING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_ZK448_WIN_S_BENCHMARK_FILE, 10)
    
    if use_comet:
        DatasetPreparation.initExperiment(projName, 'Biolip_A')
    DatasetParams.LABEL_COLUMNS = ['any_interface']
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_BIOLIP_WIN_A_TRAINING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_BIOLIP_WIN_A_TESTING_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateProtlenFreq(DatasetParams.PREPARED_ZK448_WIN_A_BENCHMARK_FILE, DatasetParams.REAL_LEN_COL_NAME)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_BIOLIP_WIN_A_TRAINING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_BIOLIP_WIN_A_TESTING_FILE, 10)
    DatasetPreparation.generateIPBs(DatasetParams.PREPARED_ZK448_WIN_A_BENCHMARK_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_BIOLIP_WIN_A_TRAINING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_BIOLIP_WIN_A_TESTING_FILE, 10)
    DatasetPreparation.generateAAIPBs(DatasetParams.PREPARED_ZK448_WIN_A_BENCHMARK_FILE, 10)
    return

def genUpdatedBiolipFile():
    DatasetPreparation.updateBiolipFile(DatasetParams.PREPARED_ZK448_WIN_A_BENCHMARK_FILE, DatasetParams.PREPARED_ZK448_WIN_P_BENCHMARK_FILE,  DatasetParams.PREPARED_ZK448_WIN_UA_BENCHMARK_FILE)
    return 

def genVennBiolipFiles():
    datasets = [DatasetParams.PREPARED_BIOLIP_WIN_P_TESTING_FILE, DatasetParams.PREPARED_BIOLIP_WIN_S_TESTING_FILE, DatasetParams.PREPARED_BIOLIP_WIN_N_TESTING_FILE]
    labels = ['BioDL_P_TE', 'BioDL_S_TE', 'BioDL_N_TE']
    DatasetPreparation.genVenn3BiolipDS(datasets, labels)
    
    datasets = [DatasetParams.PREPARED_ZK448_WIN_P_BENCHMARK_FILE, DatasetParams.PREPARED_ZK448_WIN_S_BENCHMARK_FILE, DatasetParams.PREPARED_ZK448_WIN_N_BENCHMARK_FILE]
    labels = ['ZK448_P_TE', 'ZK448_S_TE', 'ZK448_N_TE']
    DatasetPreparation.genVenn3BiolipDS(datasets, labels)
    
    datasets = [DatasetParams.PREPARED_BIOLIP_WIN_P_TRAINING_FILE, DatasetParams.PREPARED_BIOLIP_WIN_S_TRAINING_FILE, DatasetParams.PREPARED_BIOLIP_WIN_N_TRAINING_FILE]
    labels = ['BioDL_P_TR', 'BioDL_S_TR', 'BioDL_N_TR']
    DatasetPreparation.genVenn3BiolipDS(datasets, labels)
    return
   
if __name__ == "__main__":
    genVennBiolipFiles()
    #genUpdatedBiolipFile()
    #genPreparedSerendipFiles(onlyTest=False) 
    #genPreparedEpitopeFiles(onlyTest=False)
    #genPreparedBiolipFiles(onlyTest=True)
    
    #DatasetPreparation.uploadExprFigures()
    #DatasetPreparation.uploadExprHtmls()
    
    #genSerendipStats(use_comet=True)
    #genEpitopeStats(use_comet=False)
    #genBiolipStats(use_comet=False)
    
    #genBiolipStatFile()
    #addBiolipStatFeatures()
    