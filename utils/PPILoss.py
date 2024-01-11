from comet_ml import Experiment
import sys
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, \
                            confusion_matrix, average_precision_score, matthews_corrcoef, accuracy_score
import numpy as np
import pandas as pd

from tensorflow.keras import backend as K
from tensorflow.keras import metrics as M
from tensorflow.keras import losses as L
import tensorflow as tf
#import tensorflow.compat.v1 as tf

from PPIDataset import DatasetParams

logger = None

class LossParams:
    CROSS_ENTROPY = 0
    MEAN_SQUARED = 1
    JACCARD = 2
    DICE = 3
    TVERSKY = 4
    YOLO = 5
    ATT = 6
    MC_RESNET = 7
    NAMES = ['CROSS_ENTROPY', 'MEAN_SQUARED', 'JACCARD', 'DICE', 'TVERSKY', 'YOLO', 'ATT', 'MC_RESNET']
    LOSS_FUN = CROSS_ENTROPY
    USE_KERAS_ENTROPY = False
    
    # used for confusion matrix, presision, recall, f1 scores; not for auc because multiple thresholds are used for determining auc.
    CLASS_THRESHOLD = 0.40 #0.30 #0.50
    
    USE_DELETE_PAD = False      #should we delete the padded part of the y_true and y_pred?
    USE_WEIGHTED_LOSS = False   #should we punish the predictor to compensate for class imbalance?
    LOSS_ONE_WEIGHT = 0.90      #punish the predictor higher if the prediction is wrong at the positions with truth value 1.
    LOSS_ZERO_WEIGHT = 0.10     #punish the predictor lower if the prediction is wrong at the positions with truth value 0.
    TRUE_PAD_CONST = 0.0
    PRED_PAD_CONST = 1.0        #1.0==no-change; 0.0==zero'fy; 0.001==acts-like-weight
    
    YOLO_THRESHOLD = 0.20 #0.40
    YOLO_ANCHORS = '1.0,1.0'
    YOLO_ANCHORS = [float(YOLO_ANCHORS.strip()) for YOLO_ANCHORS in YOLO_ANCHORS.split(',')]
    YOLO_SCALE_NOOB, YOLO_SCALE_CONF, YOLO_SCALE_COOR, YOLO_SCALE_CLASS = 0.5, 5.0, 5.0, 1.0

    @classmethod
    def getLossFunName(cls):
        return cls.NAMES[cls.LOSS_FUN]
        
    @classmethod
    def setLossFun(cls, lossFunParam):
        cls.LOSS_FUN = lossFunParam
    
    @classmethod
    def getClassThreshold(cls):
        return cls.CLASS_THRESHOLD
    
    @classmethod
    def setClassThreshold(cls, classThresholdParam):
        cls.CLASS_THRESHOLD = classThresholdParam
    
class PPILossCls(object):
    @classmethod
    def setLogger(cls, loggerParam):
        global logger
        logger = loggerParam
    
    @classmethod
    def flattenTargetTensor(cls, y_true, y_pred):
        # (16, 1024, 1)
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)

        return y_true_f, y_pred_f
    
    @classmethod
    def maskPadTargetTensor(cls, y_true, y_pred):
        # This method is only used for validation-set. In case USE_VAR_BATCH_INPUT, we don't use padding for the validation-set.
        #print('@@@@@@@@ y_true.shape: ', y_true.shape)
        if DatasetParams.USE_VAR_BATCH_INPUT:
            # no padding was used. So, just flatten the tensors for calculating the metrics.
            y_true, y_pred = cls.flattenTargetTensor(y_true, y_pred)
        else:
            # padding was used. 'deletePadding' returns always a flattend list.
            y_true, y_pred = cls.deletePadding(y_true, y_pred)
            
        return y_true, y_pred
    
    @classmethod
    def aucMetric(cls, y_true, y_pred):
        #print(type(y_true)) #numpy.ndarray
        # y_true.dtype = float64 (0.0 or 1.0) | y_pred.dtype = float64 (0.12389 or 0.562390)
        #print(y_pred)
        #y_pred = np.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # to overcome: ValueError: Input contains NaN, infinity or a value too large for dtype('float32')
        #col_mean = np.nanmean(y_pred, axis=0)
        #y_pred = np.nan_to_num(y_pred, nan=0.0, copy=True, posinf=1.0, neginf=0.0) # gives other problems
        #y_pred = np.nan_to_num(y_pred, nan=col_mean, copy=True, posinf=col_mean, neginf=col_mean)
        
        auc = roc_auc_score(y_true, y_pred, average ='weighted')
        
        return auc
    
    @classmethod
    def apMetric(cls, y_true, y_pred):
        ap = average_precision_score(y_true, y_pred, average ='weighted')
        
        return ap
    
    @classmethod
    def mccMetric(cls, y_true, y_pred):
        y_pred = y_pred >= LossParams.CLASS_THRESHOLD
        mcc = matthews_corrcoef(y_true, y_pred)
        
        return mcc
    
    @classmethod
    def prfMetric(cls, y_true, y_pred):
        y_pred = y_pred >= LossParams.CLASS_THRESHOLD
        presScore, recallScore, f1Score, _  = precision_recall_fscore_support(y_true, y_pred, average ='binary')

        return presScore, recallScore, f1Score
    
    @classmethod
    def confMetric(cls, y_true, y_pred, testingLabel):
        y_pred = y_pred >= LossParams.CLASS_THRESHOLD
        # y_true.dtype = float64 (0.0 or 1.0) | y_pred.dtype = bool (False or True)
        confusion  = confusion_matrix(y_true, y_pred)
        logger.info('confusion_matrix: [actual_neg=[TN, FP]; actual_pos=[FN, TP]]')
        logger.info(confusion)
        
        tn, fp, fn, tp = confusion.ravel()
        specificity = tn / (tn+fp)
        
        return specificity
    
    @classmethod
    def accMetric(cls, y_true, y_pred):
        '''
        # Use the prints to check for the error: Input contains NaN, infinity or a value too large for dtype('float32').
        print("11111111 %%%%%%%%%%%%%% np.argwhere y_true %%%%%%%%%%%", np.argwhere(np.isnan(y_true)))
        print("22222222 %%%%%%%%%%%%%% np.argwhere y_true %%%%%%%%%%%", np.argwhere(np.isnan(y_pred)))
        #'''
        
        y_pred = y_pred >= LossParams.CLASS_THRESHOLD
        acc = accuracy_score(y_true, y_pred)
        #acc = M.binary_accuracy(y_true, y_pred)
        
        return acc
    
    @classmethod
    def calculateThreshold(cls, y_true, y_pred):
        def youdenCutoff(y_true, y_pred):
            # sensitivity (recall) = tpr
            # specificity          = tnr = 1-fpr
            # Youden's index (J)   = sensitivity + specificity - 1 = tpr + 1 - fpr - 1 = tpr - fpr
            
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            j_scores = tpr - fpr
            j_ordered = sorted(zip(j_scores,thresholds))
            return j_ordered[-1][1]
        
        def simpleCutoff(y_true, y_pred):
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            # for zk-448 having #amino-acids 116500: (26734,) tpr.shape:  (26734,) thresholds.shape:  (26734,)
            #print("fpr.shape: ", fpr.shape, "tpr.shape: ", tpr.shape, "thresholds.shape: ", thresholds.shape,)     
            optimal_idx = np.argmax(np.abs(tpr - fpr))      #14909
            optimal_threshold = thresholds[optimal_idx]     #0.12253821
            return optimal_threshold
        
        def equalCutoff(y_true, y_pred):
            # cutoff is calculated based on: number of actual positives == number of predicted positives
            numActualPoses = np.count_nonzero(y_true)
            #print('numActualPoses: ', numActualPoses)
            sortedPreds = np.sort(y_pred)
            #print('numPredicted: ', sortedPreds.size, ' \ y_pred.shape', y_pred.shape)
            thresholdInd = sortedPreds.size - numActualPoses
            thresholdVal = sortedPreds[thresholdInd]
            #print('thresholdInd: ', thresholdInd, ' | thresholdVal: ', thresholdVal)   
            return thresholdVal  
        
        def optimalCutoff(y_true, y_pred):
            # The optimal cut off would be where tpr is high and fpr is low
            # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
            fpr, tpr, thresholds = roc_curve(y_true, y_pred)
            i = np.arange(len(tpr)) 
            #print('number of thresholds: ', i)
            roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(thresholds, index=i)})
            #print(roc)
            roc_t = roc.ix[(roc.tf-0).abs().argsort()[:1]]
            #print('roc_t: ', roc_t)
            cutoffList = list(roc_t['threshold'])
            #print('cutoffList: ', cutoffList)
            cutoff = cutoffList[0]
            return cutoff
        
        #cutoff = youdenCutoff(y_true, y_pred)
        #cutoff = simpleCutoff(y_true, y_pred)
        #cutoff = optimalCutoff(y_true, y_pred) 
        cutoff = equalCutoff(y_true, y_pred)
        logger.info('The best cut-off value is: {0:0.2f}'.format(cutoff))
        
        return cutoff
    
    @classmethod
    def logTrainingMetrics(cls, epoch, loss, valLoss, y_true, y_pred):
        #print(y_true.dtype, '  #', y_true.shape, '  || ', y_pred.dtype, '  #', y_pred.shape)
        if LossParams.LOSS_FUN == LossParams.YOLO: 
            y_true, y_pred = cls.decodeYoloLabels(y_true, y_pred)
        elif LossParams.LOSS_FUN == LossParams.ATT: 
            y_true, y_pred = cls.decodeAttLabels(y_true, y_pred)
        elif LossParams.LOSS_FUN == LossParams.MC_RESNET: 
            y_true, y_pred = cls.decodeMCResnetLabels(y_true, y_pred)    
        else:
            y_true, y_pred = cls.maskPadTargetTensor(y_true, y_pred)
       
        #specScore = cls.confMetric(y_true, y_pred, DatasetParams.EXPR_DATASET_LABEL)
        valAcc = cls.accMetric(y_true, y_pred)
        valAuc = cls.aucMetric(y_true, y_pred)
        
        #logger.info("Epoch: {:02d} Loss: {:.4f} valLoss: {:.4f} specScore: {:.2%} AUC: {:.2%}".format(epoch, loss, valLoss, specScore, valAuc))
        logger.info("Epoch: {:02d} Loss: {:.4f} valLoss: {:.4f} valAcc: {:.2%} AUC: {:.2%}".format(epoch, loss, valLoss, valAcc, valAuc))
        
        if DatasetParams.COMET_EXPERIMENT is not None:
            metrics = {
                "loss": loss,
                "val_loss": valLoss,
                "val_acc": valAcc,
                "val_auc": valAuc,
            }
            DatasetParams.COMET_EXPERIMENT.log_metrics(dic=metrics, epoch=epoch, step=epoch)
        
        return valAuc
    
    @classmethod
    def logTestingMetrics(cls, y_true, y_pred, testingLabel):
        #float64   # (24, 97, 64)   ||  float64   # (24, 97, 64)
        #print(y_true.dtype, '  #', y_true.shape, '  || ', y_pred.dtype, '  #', y_pred.shape)
        if LossParams.LOSS_FUN == LossParams.YOLO: 
            y_true, y_pred = cls.decodeYoloLabels(y_true, y_pred)
        elif LossParams.LOSS_FUN == LossParams.ATT: 
            y_true, y_pred = cls.decodeAttLabels(y_true, y_pred)
        elif LossParams.LOSS_FUN == LossParams.MC_RESNET: 
            y_true, y_pred = cls.decodeMCResnetLabels(y_true, y_pred)    
        else:
            y_true, y_pred = cls.maskPadTargetTensor(y_true, y_pred)
        
        if DatasetParams.USE_USERDS_EVAL == True:
            return y_true, y_pred
        
        #print("y_true: ", y_true[0:1000])
        #print("y_pred: ", y_pred[0:1000])
        
        valAuc = cls.aucMetric(y_true, y_pred)
        valAp = cls.apMetric(y_true, y_pred)
        
        # Dynamically determine the class threshold.
        LossParams.CLASS_THRESHOLD = cls.calculateThreshold(y_true, y_pred)
        #y_pred_bin = y_pred >= LossParams.CLASS_THRESHOLD 
        #print('===========', np.count_nonzero(y_pred_bin))
        
        valAcc = cls.accMetric(y_true, y_pred)
        #valPPIM = cls.ppiMetric(y_true, y_pred)
        specScore = cls.confMetric(y_true, y_pred, testingLabel)
        presScore, recallScore, f1Score = cls.prfMetric(y_true, y_pred)
        mccScore = cls.mccMetric(y_true, y_pred)
        logger.info("ValAcc: {:.2%} specScore: {:.2%} presScore: {:.2%} recallScore: {:.2%} F1Score: {:.2%} MCC: {:.2%} AUC: {:.2%} AP: {:.2%}".format(
                    valAcc, specScore, presScore, recallScore, f1Score, mccScore, valAuc, valAp))
        
        if DatasetParams.COMET_EXPERIMENT is not None:
            metrics = {
                "acc_" + testingLabel: valAcc,
                "specificity_" + testingLabel: specScore,
                "auc_" + testingLabel: valAuc,
                "ap_" + testingLabel: valAp,
                "precision_" + testingLabel: presScore,
                "recall_" + testingLabel: recallScore,
                "f1score_" + testingLabel: f1Score,
                "mccscore_" + testingLabel: mccScore,
            }
            DatasetParams.COMET_EXPERIMENT.log_metrics(dic=metrics)
        
        return y_true, y_pred
    
    @classmethod
    def jacMetric(cls, y_true, y_pred):
        smooth = 100.0
        intersection = K.sum(K.abs(y_true * y_pred))
        union = K.sum(K.abs(y_true) + K.abs(y_pred))
        jac = (intersection + smooth) / (union - intersection + smooth)
        
        return jac
    
    @classmethod
    def jacLoss(cls, y_true, y_pred):
        smooth = 100.0
        jac = cls.jacMetric(y_true, y_pred)
        
        return (1 - jac) * smooth
    
    @classmethod
    def diceMetric(cls, y_true, y_pred):
        smooth = 1.0
        intersection = K.sum(K.abs(y_true * y_pred))
        union = K.sum(K.abs(y_true) + K.abs(y_pred))
        dice = 2.0 * (intersection + smooth) / (union + smooth)
        
        return dice
    
    @classmethod
    def diceLoss(cls, y_true, y_pred):
        dice = cls.diceMetric(y_true, y_pred) 
        #print(dice)
        #return K.log(dice)
        #return dice
        return 1 - dice
    
    @classmethod
    def tverskyMetric(cls, y_true, y_pred):
        # weights of 0' and 1' classes.
        #alpha = 0.3
        #beta = 0.7
        alpha = 0.9
        beta = 0.2
        smooth = 1e-10
    
        truepos = K.sum(y_true * y_pred)
        fp_and_fn = alpha * K.sum(y_pred * (1 - y_true)) + beta * K.sum((1 - y_pred) * y_true)
        tversky = (truepos + smooth) / ((truepos + smooth) + fp_and_fn)
        
        return tversky
    
    @classmethod
    def tverskyLoss(cls, y_true, y_pred):
        tversky = cls.tverskyMetric(y_true, y_pred)
        
        return 1 - tversky
    
    @classmethod
    def decodeMCResnetLabels(cls, att_y_true, att_y_pred):
        def printAttData(nt, ht):
            print("att_y_true.shape: ", att_y_true.shape)     #(24, 97, 64)
            print("att_y_pred.shape: ", att_y_pred.shape)     #(24, 97, 64)
            return
        
        #printAttData()
        nd,hd,cd = att_y_true.shape
        y_true = np.array([], DatasetParams.FLOAT_TYPE)
        y_pred = np.array([], DatasetParams.FLOAT_TYPE)
        for n in range(nd):
            # Ignore predictions for positions beyond the protLen. 
            hd = DatasetParams.REAL_VAL_PROT_LENS[n] // DatasetParams.MC_RESNET_PAT_LEN
            rem = DatasetParams.REAL_VAL_PROT_LENS[n] % DatasetParams.MC_RESNET_PAT_LEN
            if rem > 0:
                hd = hd + 1
            tmpLabelShape = (hd, DatasetParams.MC_RESNET_PAT_LEN) 
            tmp_y_true = np.zeros(tmpLabelShape, DatasetParams.FLOAT_TYPE)
            tmp_y_pred = np.zeros(tmpLabelShape, DatasetParams.FLOAT_TYPE) 
            for h in range(hd):               # we skip the last one because it's the END indicator.
                trueClass = att_y_true[n,h,:]
                trueClassId = np.argmax(trueClass)
                predClass = att_y_pred[n,h,:]
                predClassId = np.argmax(predClass)
                trueGridCell = DatasetParams.MC_RESNET_PATTERNS[trueClassId] 
                tmp_y_true[h,:] = trueGridCell 
                predGridCell = DatasetParams.MC_RESNET_PATTERNS[predClassId] 
                tmp_y_pred[h,:] = predGridCell
                if False and np.count_nonzero(y_true[n,h,:]) != 0:
                    print("trueClassId: ", trueClassId) 
                    print("predClassId: ", predClassId) 
            flat_tmp_y_true = K.flatten(tmp_y_true)
            flat_tmp_y_pred = K.flatten(tmp_y_pred)
            if rem > 0:
                indx = DatasetParams.MC_RESNET_PAT_LEN - rem
                flat_tmp_y_true = np.delete(flat_tmp_y_true, np.s_[-indx:])
                flat_tmp_y_pred = np.delete(flat_tmp_y_pred, np.s_[-indx:])
            y_true = np.concatenate([y_true, flat_tmp_y_true])
            y_pred = np.concatenate([y_pred, flat_tmp_y_pred])
        return y_true, y_pred
    
    @classmethod
    def decodeAttLabels(cls, att_y_true, att_y_pred):
        def printAttData():
            print("att_y_true.shape: ", att_y_true.shape)     #(24, 97, 64)
            print("att_y_pred.shape: ", att_y_pred.shape)     #(24, 97, 64)
            return
        
        #printAttData()
        nd,hd,cd = att_y_true.shape
        y_true = np.array([], DatasetParams.FLOAT_TYPE)
        y_pred = np.array([], DatasetParams.FLOAT_TYPE)
        for n in range(nd):
            # Ignore predictions for positions beyond the protLen. 
            hd = DatasetParams.REAL_VAL_PROT_LENS[n] // DatasetParams.ATT_PAT_LEN
            rem = DatasetParams.REAL_VAL_PROT_LENS[n] % DatasetParams.ATT_PAT_LEN
            if rem > 0:
                hd = hd + 1
            tmpLabelShape = (hd, DatasetParams.ATT_PAT_LEN) 
            tmp_y_true = np.zeros(tmpLabelShape, DatasetParams.FLOAT_TYPE)
            tmp_y_pred = np.zeros(tmpLabelShape, DatasetParams.FLOAT_TYPE)    
            for h in range(hd):               # we skip the last one because it's the END indicator.
                trueClass = att_y_true[n,h,:]
                trueClassId = np.argmax(trueClass)
                predClass = att_y_pred[n,h,:]
                predClassId = np.argmax(predClass)
                trueGridCell = DatasetParams.ATT_PATTERNS[trueClassId] 
                trueGridCell = np.delete(trueGridCell,0)
                tmp_y_true[h,:] = trueGridCell 
                predGridCell = DatasetParams.ATT_PATTERNS[predClassId] 
                predGridCell = np.delete(predGridCell,0)
                tmp_y_pred[h,:] = predGridCell
                if False and np.count_nonzero(tmp_y_true[h,:]) != 0:
                    print("trueClassId: ", trueClassId)     
                    print("predClassId: ", predClassId) 
                    #print("y_true: ", tmp_y_true[h,:])
                    #print("y_pred: ", tmp_y_pred[h,:])
            flat_tmp_y_true = K.flatten(tmp_y_true)
            flat_tmp_y_pred = K.flatten(tmp_y_pred)
            if rem > 0:
                indx = DatasetParams.ATT_PAT_LEN - rem
                flat_tmp_y_true = np.delete(flat_tmp_y_true, np.s_[-indx:])
                flat_tmp_y_pred = np.delete(flat_tmp_y_pred, np.s_[-indx:])
            y_true = np.concatenate([y_true, flat_tmp_y_true])
            y_pred = np.concatenate([y_pred, flat_tmp_y_pred])

        return y_true, y_pred
    
    @classmethod
    def attLoss(cls, y_true, y_pred):
        #y_true, y_pred = cls.decodeAttLabels(y_true, y_pred)
        #loss = L.binary_crossentropy(y_true, y_pred)
        loss = L.categorical_crossentropy(y_true, y_pred)
        return loss
    
    @classmethod
    def mcResnetLoss(cls, y_true, y_pred):
        #y_true, y_pred = cls.decodeAttLabels(y_true, y_pred)
        #loss = L.binary_crossentropy(y_true, y_pred)
        loss = L.categorical_crossentropy(y_true, y_pred)
        return loss
    
    @classmethod
    def decodeYoloLabels(cls, yolo_y_true, yolo_y_pred):
        def printYoloData(y_true, y_pred):
            print(y_true.shape)
            print(y_pred.shape)
            print(y_true[0,1,2])  # [[1. 1.][0. 0.]]  
            print(y_pred[0,1,2])  # [[0. 0.][0. 0.]]  
            print(y_true[0,1,1])  # [[0. 1.][0. 0.]]
            print(y_pred[0,1,1])  # [[0. 0.][0. 0.]]
        
        #yolo_y_true: float64-(56, 12, 12, 1, 21)   ||  yolo_y_pred:float64(56, 12, 12, 1, 21)
        nd,hd,wd,bd,cd = yolo_y_true.shape
        # (56, 12, 12, 2, 2)
        tmpLabelShape = (nd, hd, wd, DatasetParams.GRID_CELL_H, DatasetParams.GRID_CELL_W)
        y_true = np.zeros(tmpLabelShape, DatasetParams.FLOAT_TYPE)
        y_pred = np.zeros(tmpLabelShape, DatasetParams.FLOAT_TYPE)
        for n in range(nd):
            for h in range(hd):
                for w in range(wd):
                    for b in range(bd):
                        trueClass = yolo_y_true[n,h,w,b,5:]
                        trueClassId = K.argmax(trueClass)
                        predClassId = 0
                        conf = K.sigmoid(yolo_y_pred[n,h,w,b,4])
                        if conf > LossParams.YOLO_THRESHOLD:
                            classProb = K.softmax(yolo_y_pred[n,h,w,b,5:]) * conf
                            predClassId = K.argmax(classProb)
                    
                    y_true[n,h,w,:] = DatasetParams.YOLO_PATTERNS[trueClassId]         
                    y_pred[n,h,w,:] = DatasetParams.YOLO_PATTERNS[predClassId] 
                    #print(y_true[n,h,w,:])
                    #print(y_pred[n,h,w,:])
        
        #printYoloData(y_true, y_pred)
        
        # (56, 24, 24, 1)
        #labelShape = (nd,) + DatasetParams.LABEL_SHAPE
        #y_true = np.reshape(y_true, labelShape)
        #y_pred = np.reshape(y_pred, labelShape)
        y_true = K.flatten(y_true)
        y_pred = K.flatten(y_pred)
        
        return y_true, y_pred
    
    @classmethod
    def yoloLoss(cls, y_true, y_pred):
        ### Adjust prediction
        # adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[:,:,:,:,:2])
        
        # adjust w and h
        pred_box_wh = tf.exp(y_pred[:,:,:,:,2:4]) * np.reshape(LossParams.YOLO_ANCHORS, [1,1,1,DatasetParams.NUM_BOXES,2])
        pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(DatasetParams.GRID_H), float(DatasetParams.GRID_W)], [1,1,1,1,2]))
        
        # adjust confidence
        pred_box_conf = tf.expand_dims(tf.sigmoid(y_pred[:, :, :, :, 4]), -1)
        
        # adjust probability
        pred_box_prob = tf.nn.softmax(y_pred[:, :, :, :, 5:])
        
        y_pred = tf.concat([pred_box_xy, pred_box_wh, pred_box_conf, pred_box_prob], 4)
        #print("Y_pred shape: {}".format(y_pred.shape))
        
        ### Adjust ground truth
        # adjust x and y
        center_xy = .5*(y_true[:,:,:,:,0:2] + y_true[:,:,:,:,2:4])
        center_xy = center_xy / np.reshape([float(DatasetParams.GRID_CELL_H), float(DatasetParams.GRID_CELL_W)], [1,1,1,1,2])
        true_box_xy = center_xy - tf.floor(center_xy)
        
        # adjust w and h
        true_box_wh = (y_true[:,:,:,:,2:4] - y_true[:,:,:,:,0:2])
        true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(DatasetParams.YOLO_NORM_H), float(DatasetParams.YOLO_NORM_W)], [1,1,1,1,2]))
        
        # adjust confidence
        pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([DatasetParams.GRID_H, DatasetParams.GRID_W], [1,1,1,1,2])
        pred_box_area = pred_tem_wh[:,:,:,:,0] * pred_tem_wh[:,:,:,:,1]
        pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
        pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh
        
        true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([DatasetParams.GRID_H, DatasetParams.GRID_W], [1,1,1,1,2])
        true_box_area = true_tem_wh[:,:,:,:,0] * true_tem_wh[:,:,:,:,1]
        true_box_ul = true_box_xy - 0.5 * true_tem_wh
        true_box_bd = true_box_xy + 0.5 * true_tem_wh
        
        intersect_ul = tf.maximum(pred_box_ul, true_box_ul) 
        intersect_br = tf.minimum(pred_box_bd, true_box_bd)
        intersect_wh = intersect_br - intersect_ul
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect_area = intersect_wh[:,:,:,:,0] * intersect_wh[:,:,:,:,1]
        
        iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
        best_box = tf.equal(iou, tf.reduce_max(iou, [3], True)) 
        best_box = tf.to_float(best_box)
        true_box_conf = tf.expand_dims(best_box * y_true[:,:,:,:,4], -1)
        
        # adjust confidence
        true_box_prob = y_true[:,:,:,:,5:]
        
        y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
        #print("Y_true shape: {}".format(y_true.shape))
        #y_true = tf.Print(y_true, [true_box_wh], message='DEBUG', summarize=30000)    
        
        ### Compute the weights
        weight_coor = tf.concat(4 * [true_box_conf], 4)
        weight_coor = LossParams.YOLO_SCALE_COOR * weight_coor
        
        weight_conf = LossParams.YOLO_SCALE_NOOB * (1. - true_box_conf) + LossParams.YOLO_SCALE_CONF * true_box_conf
        
        weight_class = tf.concat(DatasetParams.NUM_CLASSES * [true_box_conf], 4) 
        weight_class = LossParams.YOLO_SCALE_CLASS * weight_class
        
        weight = tf.concat([weight_coor, weight_conf, weight_class], 4)
        #print(" shape: {}".format(weight.shape))
        
        ### Finalize the loss
        loss = tf.pow(y_pred - y_true, 2)
        loss = loss * weight
        loss = tf.reshape(loss, [-1, DatasetParams.GRID_H*DatasetParams.GRID_W*DatasetParams.NUM_BOXES*(4 + 1 + DatasetParams.NUM_CLASSES)])
        loss = tf.reduce_sum(loss, 1)
        loss = .5 * tf.reduce_mean(loss)
        #print(loss)
        
        return loss
    
    @classmethod
    def ppiMetric(cls, y_true, y_pred):
        # y_pred.shape = (16, 1024, 1) of probablities; dtype=float64; it's the same for y_true except for that the values of y_true are 1.0 or 0.0.
        # This is actually very strange becasue de type of y_true was "int32" and not float64.
    
        if LossParams.LOSS_FUN == LossParams.CROSS_ENTROPY:
            metric = M.binary_accuracy(y_true, y_pred)
        elif LossParams.LOSS_FUN == LossParams.MEAN_SQUARED: 
            metric = M.mean_squared_error(y_true, y_pred)
        elif LossParams.LOSS_FUN == LossParams.JACCARD:      
            metric = cls.jacMetric(y_true, y_pred)  
        elif LossParams.LOSS_FUN == LossParams.DICE:      
            metric = cls.diceMetric(y_true, y_pred)   
        elif LossParams.LOSS_FUN == LossParams.TVERSKY:      
            metric = cls.tverskyMetric(y_true, y_pred)     
        elif LossParams.LOSS_FUN == LossParams.YOLO:    
            metric = M.binary_accuracy(y_true, y_pred)   
        elif LossParams.LOSS_FUN == LossParams.ATT:    
            metric = M.binary_accuracy(y_true, y_pred) 
        elif LossParams.LOSS_FUN == LossParams.MC_RESNET:    
            metric = M.binary_accuracy(y_true, y_pred)       
        
        return metric
    
    @classmethod
    def changePadding(cls, y_true, y_pred):
        y_true = tf.where(tf.not_equal(y_true,DatasetParams.LABEL_PAD_CONST), y_true, LossParams.TRUE_PAD_CONST)
        y_pred = tf.where(tf.not_equal(y_true,DatasetParams.LABEL_PAD_CONST), y_pred, tf.multiply(y_pred,LossParams.PRED_PAD_CONST))
        return y_true,y_pred
    
    @classmethod
    def deletePadding(cls, y_true, y_pred):
        mask = K.not_equal(y_true, DatasetParams.LABEL_PAD_CONST)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        return y_true,y_pred
    
    @classmethod
    def binCrossEntropyLoss(cls, y_true, y_pred):
        """
        homo=F1Score: 47.12% MCC: 31.66% AUC: 74.68% vs. F1Score: 47.93% MCC: 32.71% AUC: 75.16% (Keras vs. own)
        hetro=F1Score: 18.13% MCC: 9.65% AUC: 64.88% vs. F1Score: 17.80% MCC: 9.29% AUC: 63.47%  (Keras vs. own)
        """
        if LossParams.USE_KERAS_ENTROPY:
            loss = K.binary_crossentropy(y_true, y_pred)
        else:
            loss = y_true * (-K.log(y_pred)) + (1 - y_true) * (-K.log(1 - y_pred))
        
        if LossParams.USE_WEIGHTED_LOSS:
            lossWeights = (y_true * LossParams.LOSS_ONE_WEIGHT) + (1.0 - y_true) * LossParams.LOSS_ZERO_WEIGHT
            loss = loss * lossWeights
        bce = K.mean(loss)
        
        return bce
    
    @classmethod
    def meanSquaredLoss(cls, y_true, y_pred):
        loss = K.square(tf.math.subtract(y_true, y_pred))
        #loss = L.MeanSquaredError(y_true, y_pred)
        if LossParams.USE_WEIGHTED_LOSS:
            lossWeights = (y_true * LossParams.LOSS_ONE_WEIGHT) + (1.0 - y_true) * LossParams.LOSS_ZERO_WEIGHT
            loss = loss * lossWeights
        mse = K.mean(loss)
        
        return mse
    
    @classmethod
    def ppiLoss(cls, y_true, y_pred):
        # Note: each element in y_pred is a data-point (all amino acids of all proteins of a batch and plus padded postions) and 
        # loss will be calculated for all these data-points. Finally, the loss-value will be the average of the loss of all data-points.
        #print("ppiLoss:before:y_true.shape: ", y_true.shape)
        if LossParams.LOSS_FUN not in [LossParams.YOLO, LossParams.ATT, LossParams.MC_RESNET]:  
            if LossParams.USE_DELETE_PAD:
                y_true, y_pred = cls.deletePadding(y_true, y_pred)
            else:
                y_true, y_pred = cls.changePadding(y_true, y_pred)     
        #print("ppiLoss:after:y_true.shape: ", y_true.shape)
        
        
        if LossParams.LOSS_FUN == LossParams.CROSS_ENTROPY:
            loss = cls.binCrossEntropyLoss(y_true, y_pred)
        elif LossParams.LOSS_FUN == LossParams.MEAN_SQUARED: 
            loss = cls.meanSquaredLoss(y_true, y_pred)   
        elif LossParams.LOSS_FUN == LossParams.JACCARD:      
            loss = cls.jacLoss(y_true, y_pred)  
        elif LossParams.LOSS_FUN == LossParams.DICE:      
            loss = cls.diceLoss(y_true, y_pred)  
        elif LossParams.LOSS_FUN == LossParams.TVERSKY:      
            loss = cls.tverskyLoss(y_true, y_pred)    
        elif LossParams.LOSS_FUN == LossParams.YOLO:     
            loss = cls.yoloLoss(y_true, y_pred) 
        elif LossParams.LOSS_FUN == LossParams.ATT:     
            loss = cls.attLoss(y_true, y_pred)
        elif LossParams.LOSS_FUN == LossParams.MC_RESNET:     
            loss = cls.mcResnetLoss(y_true, y_pred)             
        #loss = M.losses.binary_crossentropy(y_true, y_pred) + jacLoss(y_true, y_pred)
        #loss = M.losses.binary_crossentropy(y_true, y_pred) - diceLoss(y_true, y_pred)
        
        return loss
    
