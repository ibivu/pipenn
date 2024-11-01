import sys

import pandas as pd

from PPIDataset import DatasetParams, DatasetPreparation

DF_FEATURE_COLUMNS = [
    # Don't change the order.
    #0:4
    'uniprot_id',    
    'sequence',
    'Rlength',
    'normalized_length',
    'any_interface', #4
    ]

NORMALIZATION_DATA = {
        #Min and Max values for LENGTH in the training data set: min=26;max=700;
        #DF_FEATURE_COLUMNS[8]: [0.0, 700.0],??
        DF_FEATURE_COLUMNS[3]: [0.0, 2050.0],
    }


def genUserDSProtBertFile():
    DatasetParams.SEQ_COLUMN_NAME = DF_FEATURE_COLUMNS[1]
    DatasetPreparation.createProtbertFile(DatasetParams.PREPARED_USERDS_FILE)
    return

def genPreparedPipennProtFiles(df_inp):
    print('Generating PIPENN file for user input ...')
    df_inp.to_csv(DatasetParams.PREPARED_USERDS_FILE, index=False) 
    return
    
def generateDFInput(prot_seqs, prot_lens, prot_ids):
    def normalizeData(df):
        colNames = NORMALIZATION_DATA.keys()
        for normColName in colNames:
            try:
                minVal = NORMALIZATION_DATA[normColName][0]
                maxVal = NORMALIZATION_DATA[normColName][1]
                #print("normColName={} min={} and max={}".format(normColName, minVal, maxVal))
                df[normColName] = (df[normColName] - minVal) / (maxVal - minVal)
                df.loc[df[normColName]>1.0, normColName] = 1.0 
            except KeyError:
                print('WARNING: AA LEN is out of range of min-max. LEN=[26,2050]')
        
        return df
    
    def repeatCols(df_inp, prot_lens):
        normColName = DF_FEATURE_COLUMNS[3]
        intColName = DF_FEATURE_COLUMNS[4]
        normColVals = df_inp[normColName]
        for i in range(len(normColVals)):
            normColVal = normColVals[i]
            prot_len = prot_lens[i]
            normColValSeq = (str(normColVal)+',') * prot_len
            normColValSeq = normColValSeq.rstrip(normColValSeq[-1])
            df_inp.loc[i, normColName] = normColValSeq
            
            intColValSeq = ('0'+',') * prot_len
            intColValSeq = intColValSeq.rstrip(intColValSeq[-1])
            df_inp.loc[i, intColName] = intColValSeq
        return df_inp
    
    def addCols(df_inp, prot_ids, prot_lens):
        id_col_name = DF_FEATURE_COLUMNS[0]
        df_inp[id_col_name] = prot_ids
        df_inp[DF_FEATURE_COLUMNS[1]] = prot_seqs
        for i in range(len(prot_ids)):
            prot_id = prot_ids[i]
            prot_len = prot_lens[i]
            df_inp.loc[df_inp[id_col_name]==prot_id, DF_FEATURE_COLUMNS[2:4]] = prot_len
        df_inp[DF_FEATURE_COLUMNS[4]] = 0
        #print(df_inp)
        return df_inp
    
    print('Making dataframe for user input ...')
    df_inp = pd.DataFrame(columns=DF_FEATURE_COLUMNS)
    df_inp = addCols(df_inp, prot_ids, prot_lens)
    df_inp = normalizeData(df_inp)
    df_inp = repeatCols(df_inp, prot_lens)
    #print(df_inp)
    return df_inp

def parseUseInput():
    from Bio import SeqIO
    
    print('Parsing user input file ...')
    prot_lens = []
    prot_ids = []
    prot_seqs = []
    for rec in SeqIO.parse(DatasetParams.USER_INPUT_FASTA_FILE, 'fasta'):
        prot_seqs.append(str(','.join(rec.seq)))
        prot_lens.append(len(rec.seq))
        prot_ids.append(rec.id)
    #print('num_prots: ', len (prot_lens), ' Protlens: ', prot_lens, ' ProtIds: ', prot_ids)
    return prot_seqs, prot_lens, prot_ids
    
def generatePipennInput():
    prot_seqs, prot_lens, prot_ids = parseUseInput()
    df_inp = generateDFInput(prot_seqs, prot_lens, prot_ids)
    genPreparedPipennProtFiles(df_inp)
    genUserDSProtBertFile()
    return

if __name__ == "__main__":
    nargs = len(sys.argv)
    if nargs == 4:
        pipennHome = sys.argv[1]
        DatasetParams.PIPENN_HOME = pipennHome
        DatasetParams.PROT_BERT_MODEL_DIR = DatasetParams.PIPENN_HOME + "/protbert"
        DatasetParams.USER_INPUT_FASTA_FILE = sys.argv[2]
        DatasetParams.USERDS_INPUT_DIR = './'
        DatasetParams.PREPARED_USERDS_FILE = sys.argv[3]
        print("PIPENN_HOME:", DatasetParams.PIPENN_HOME,
              "USER_INPUT_FASTA_FILE:", DatasetParams.USER_INPUT_FASTA_FILE,
              " | PREPARED_USERDS_FILE: ", DatasetParams.PREPARED_USERDS_FILE)
    else:
        print("$$$$$$$$ No proper arguments passed - nargs: ", nargs)
    
    #pd.set_option('display.max_columns', None)
    #pd.set_option('display.max_rows', None)   
    generatePipennInput()
    
