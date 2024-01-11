import sys, os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve, \
                            confusion_matrix, accuracy_score, average_precision_score, matthews_corrcoef

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from audioop import avg

from PPIDataset import DatasetParams

class PPIPredMericsCls(object):   
    NUM_DIG = 3
    
    @classmethod
    def calcMetrics(cls, y_trues, y_preds, fpr, tpr, thresholds):
        def simpleCutoff():
            # for zk-448 having #amino-acids 116500: (26734,) tpr.shape:  (26734,) thresholds.shape:  (26734,)
            #print("fpr.shape: ", fpr.shape, "tpr.shape: ", tpr.shape, "thresholds.shape: ", thresholds.shape,)     
            #cutoffInd = np.argmax(np.abs(tpr - fpr))      #14909
            #cutoff = thresholds[cutoffInd]     #0.12253821
            cutoff = 0.500
            return cutoff
        
        def optimalCutoff():
            # The optimal cut off would be where tpr is high and fpr is low
            # tpr - (1-fpr) is zero or near to zero is the optimal cut off point
            i = np.arange(len(tpr)) 
            #print('number of thresholds: ', i)
            roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(thresholds, index=i)})
            #print(roc)
            roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
            #print('roc_t: ', roc_t)
            cutoffList = list(roc_t['threshold'])
            #print('cutoffList: ', cutoffList)
            cutoff = cutoffList[0]
            return cutoff
        
        def equalCutoff():
            # cutoff is calculated based on: number of actual positives == number of predicted positives
            numActualPoses = np.count_nonzero(y_trues)
            #print('numActualPoses: ', numActualPoses)
            sortedPreds = np.sort(y_preds)
            #print('numPredicted: ', sortedPreds.size, ' \ y_pred.shape', y_pred.shape)
            cutoffInd = sortedPreds.size - numActualPoses
            cutoff = sortedPreds[cutoffInd]
            #print('cutoffInd: ', cutoffInd, ' | cutoff: ', cutoff)   
            return cutoff
        
        def youdenCutoff():
            # sensitivity (recall) = tpr
            # specificity          = tnr = 1-fpr
            # Youden's index (J)   = sensitivity + specificity - 1 = tpr + 1 - fpr - 1 = tpr - fpr (distance between curve and the chance line
            j_scores = tpr - fpr
            j_ordered = sorted(zip(j_scores,thresholds))
            return j_ordered[-1][1]
        
        def consScoreDataDict(cutoffMethod, cutoff):
            cutoffPreds = y_preds >= cutoff
            accScore = accuracy_score(y_trues, cutoffPreds)
            confusion  = confusion_matrix(y_trues, cutoffPreds)
            tn, fp, fn, tp = confusion.ravel()
            specScore = tn / (tn+fp)
            presScore, recallScore, f1Score, _  = precision_recall_fscore_support(y_trues, cutoffPreds, average ='binary')
            mccScore  = matthews_corrcoef(y_trues, cutoffPreds)
            scoreDataDict = {
                'cutoff_method': cutoffMethod,
                'cutoff': round(cutoff, cls.NUM_DIG),
                'confusion[TN,FP,FN,TP]': confusion,
                'accuracy': round(accScore, cls.NUM_DIG), 
                'specificity': round(specScore, cls.NUM_DIG),
                'precision': round(presScore, cls.NUM_DIG),
                'recall': round(recallScore, cls.NUM_DIG),
                'f1': round(f1Score, cls.NUM_DIG),
                'mcc': round(mccScore, cls.NUM_DIG),
            }
            return scoreDataDict
        
        scoreDataDicts = []
        scoreDataDicts.append(consScoreDataDict('SIMPLE', simpleCutoff()))
        scoreDataDicts.append(consScoreDataDict('YOUDEN', youdenCutoff()))
        scoreDataDicts.append(consScoreDataDict('OPTIMAL', optimalCutoff()))
        scoreDataDicts.append(consScoreDataDict('EQUAL', equalCutoff()))        
        return scoreDataDicts
    
    @classmethod 
    def getMetrics(cls, y_trues, y_preds):
        #y_trues == y_preds == [ [preds-for-prot-1], ..., [preds-for-prot-n] ]
        y_trues, y_preds = np.concatenate(y_trues), np.concatenate(y_preds)
        aucScore = roc_auc_score(y_trues, y_preds, average ='weighted')
        apScore = average_precision_score(y_trues, y_preds, average ='weighted')
        precisions, recalls, thsPR = precision_recall_curve(y_trues, y_preds)
        fpr, tpr, thsROC = roc_curve(y_trues, y_preds)
        scoreDataDicts = cls.calcMetrics(y_trues, y_preds, fpr, tpr, thsROC)
        curveDataDict = {
            'thsROC': np.round(thsROC, cls.NUM_DIG),
            'thsPR': np.round(thsPR, cls.NUM_DIG),
            'truePosRate': np.round(tpr, cls.NUM_DIG),
            'falsePosRate': np.round(fpr, cls.NUM_DIG),
            'aucScore': round(aucScore, cls.NUM_DIG),
            'apScore': round(apScore, cls.NUM_DIG),
            'precisions': np.round(precisions, cls.NUM_DIG),
            'recalls': np.round(recalls, cls.NUM_DIG),
        }
       
        return scoreDataDicts, curveDataDict
    
    @classmethod
    def consFigMetrics(cls, y_trues, y_preds):
        # y_trues = [(120,), (480,), (56,), ..., (120,)]
        def createSubplots(auc, ap):
            figPlot = make_subplots(rows=2, cols=2, shared_xaxes=False, 
                                    vertical_spacing=0.1, 
                                    subplot_titles=('ROC Curve with AUC={}'.format(auc), 'Recall Precision Curve AP={}'.format(ap)),
                                    specs=[ [{}, {}],
                                            [{'colspan':2, 'type':'table'}, None]
                                          ],
                                    row_heights=[0.80, 0.20],
                                    )
            figPlot.update_layout(dict(
                       paper_bgcolor="LightSteelBlue",
                       height=710,
                       width=1300,
                       autosize = False,
                       margin_b=10,
                       margin_t=40,
                      ))
            return figPlot
        
        def plotRocCurve(figPlot, fpr, tpr, thresholds):
            trace = go.Scatter(x=[0,1],y=[0,1],hoverinfo='x+y',showlegend=False,marker_color='lightgreen')
            figPlot.add_trace(trace, 1, 1)
            trace = go.Scatter(x=fpr,y=tpr,hoverinfo='x+y+text',showlegend=False,text=thresholds,mode='lines')
            figPlot.add_trace(trace, 1, 1)
            figPlot.update_xaxes(patch={'title': '1 - Specificity (FPR)'}, row=1,col=1)
            figPlot.update_yaxes(patch={'title': 'Sensitivity (TPR)'}, row=1,col=1)
            return figPlot
        
        def plotPRCurve(figPlot, recalls, precisions, thresholds):
            allAAs = np.concatenate(y_trues)
            #print('$$$$$$$$$$$$$$$$$$$ allAAs.shape: ', allAAs.shape)
            numPoses = np.count_nonzero(allAAs)
            endPoint = numPoses / allAAs.size   
            trace = go.Scatter(x=[0,1],y=[endPoint,endPoint],hoverinfo='x+y',showlegend=False,marker_color='lightgreen')
            figPlot.add_trace(trace, 1, 2)
            #trace = go.Scatter(x=recalls,y=precisions,hoverinfo='x+y+text',showlegend=False,text=thresholds,mode='lines')
            trace = go.Scatter(x=recalls[:-1],y=precisions[:-1],hoverinfo='x+y+text',showlegend=False,text=thresholds[:-1],mode='lines')
            figPlot.add_trace(trace, 1, 2)
            figPlot.update_xaxes(patch={'title': 'Recall'}, row=1,col=2)
            figPlot.update_yaxes(patch={'title': 'Precision'}, row=1,col=2)
            return figPlot
        
        def plotScoreTable(figPlot, scoreDataDicts):
            colWidths =[14,8,30,8,8,8,8,8,8] 
            tableHeaders = list(scoreDataDicts[0].keys())
            tableVals = []
            for cm in range(len(scoreDataDicts)):
                tableVals.append(list(scoreDataDicts[cm].values()))
            tableVals = [list(i) for i in zip(*tableVals)]
            
            trace = go.Table(header=dict(values=tableHeaders),columnwidth=colWidths,cells=dict(values=tableVals))
            figPlot.add_trace(trace, 2, 1)
            return figPlot
        
        scoreDataDicts, curveDataDict = cls.getMetrics(y_trues, y_preds)
        figPlot = createSubplots(curveDataDict['aucScore'], curveDataDict['apScore'])
        figPlot = plotRocCurve(figPlot, curveDataDict['falsePosRate'], curveDataDict['truePosRate'], curveDataDict['thsROC'])
        figPlot = plotPRCurve(figPlot, curveDataDict['recalls'], curveDataDict['precisions'], curveDataDict['thsPR'])
        figPlot = plotScoreTable(figPlot, scoreDataDicts)
        return figPlot
    
    @classmethod
    def consStackedFigMetrics(cls, y_trues, y_stacked_preds, modelsNames, datasetName):
        # y_trues.shape= (84941,) and y_stacked_preds.shape=(84941, 1, 7)
        def createStackedSubplots():
            figPlot = make_subplots(rows=1, cols=2,
                                    subplot_titles=['ROC Curves for @{}@'.format(datasetName),
                                                    'PR Curves for @{}@'.format(datasetName)],
                                    ) 
            figPlot.update_layout(dict(
                               #paper_bgcolor="rgba(0,0,0,0)",
                               height=600,
                               width=1300,
                               autosize = False,
                               #legend=dict(
                                  #yanchor="top",
                                  #y=0.45,
                                  #xanchor="right",
                                  #x=0.99)
                              ))
            return figPlot
                
        def plotStackedRocCurve(figPlot, fpr, tpr, thresholds, modelName, modelColor):
            trace = go.Scatter(x=fpr,y=tpr,hoverinfo='x+y+text',text=thresholds,mode='lines',name=modelName,
                               showlegend=False,marker_color=modelColor)
            figPlot.add_trace(trace,1,1)
            figPlot.update_xaxes(patch={'title': '1 - Specificity (FPR)'},row=1,col=1)
            figPlot.update_yaxes(patch={'title': 'Sensitivity (TPR)'},row=1,col=1)
            return figPlot
        
        def plotStackedPRCurve(figPlot, recalls, precisions, thresholds, modelName, modelColor):
            trace = go.Scatter(x=recalls[:-1],y=precisions[:-1],hoverinfo='x+y+text',text=thresholds[:-1],mode='lines',name=modelName,
                               marker_color=modelColor)
            figPlot.add_trace(trace, 1, 2)
            figPlot.update_xaxes(patch={'title': 'Recall'}, row=1,col=2)
            figPlot.update_yaxes(patch={'title': 'Precision'}, row=1,col=2)
            return figPlot
        
        modelColors = plotly.colors.DEFAULT_PLOTLY_COLORS
        figPlot = createStackedSubplots()
        figPlot = plotStackedRocCurve(figPlot, [0,1], [0,1], [], 'Random', modelColors[0])
        allAAs = y_trues
        #print('$$$$$$$$$$$$$$$$$$$ allAAs.shape: ', allAAs.shape)
        numPoses = np.count_nonzero(allAAs)
        endPoint = numPoses / allAAs.shape[0]   
        figPlot = plotStackedPRCurve(figPlot, [0,1,5], [endPoint,endPoint,5], [], 'Random', modelColors[0])   #5 is added to compensate recalls[:-1]. 
        numModels = len(modelsNames)
        #getmetrics expects 'list of flat_y_trues and y_preds'. So we must pack them in lists.
        y_trues = [y_trues]
        for i in range(numModels):
            y_preds = y_stacked_preds[:,:,i]
            y_preds = [y_preds.flatten()] #getmetrics expects 'list of flat_y_trues and y_preds'. So we must pack them in lists.
            scoreDataDicts, curveDataDict = cls.getMetrics(y_trues, y_preds)
            figPlot = plotStackedRocCurve(figPlot, curveDataDict['falsePosRate'], curveDataDict['truePosRate'], curveDataDict['thsROC'], 
                                          modelsNames[i], modelColors[i])
            figPlot = plotStackedPRCurve(figPlot, curveDataDict['recalls'], curveDataDict['precisions'], curveDataDict['thsPR'], 
                                         modelsNames[i], modelColors[i])
        
        return figPlot
    
    @classmethod
    def consCombinedPerProtFigMetrics(cls, metricsPerModel, modelsNames, datasetName, plotType):  
        # metricsPerModel = [pandas-df-model-1, ..., pandas-df-model-n]
        
        def getProtInfoList(protIds, protLens):
            numProts = len(protIds)
            protInfoList = []
            for i in range(0, numProts):
                protId = protIds[i]
                protLen = str(protLens[i])
                protInfo = protId + ':' + protLen
                protInfoList.append(protInfo)
            return protInfoList
                
        def plotCombinedViolinMCC(numModels, protInfoList, modelColors):
            def createViolinMCCSubplots():
                figPlot = make_subplots(rows=1, cols=1,
                                        subplot_titles=['MCC Violins for @{}@'.format(datasetName),
                                                        ],
                                        ) 
                figPlot.update_layout(dict(
                                    #paper_bgcolor="rgba(0,0,0,0)",
                                    #height=600,
                                    #width=1300,
                                    autosize = False,
                                    hovermode="x",
                                  ))
                return figPlot
            
            def plotViolinMCC(figPlot, modelMCCs, modelName, protInfoList, modelColor):
                trace = go.Violin(x=modelMCCs,hoverinfo='x+y+text',text=protInfoList,orientation='h',name=modelName,
                                   showlegend=False,box_visible=True,meanline_visible=True,marker_color=modelColor,
                                   #side='positive', 
                                   points='all',
                                   )
                figPlot.add_trace(trace,1,1)
                figPlot.update_xaxes(patch={'title': 'MCC'},row=1,col=1)
                figPlot.update_yaxes(patch={'title': 'Model'},row=1,col=1)
                return figPlot
            
            figPlot = createViolinMCCSubplots()
            for i in range(numModels):
                modelMetricsDF = metricsPerModel[i]
                modelMCCs = modelMetricsDF['prot_mcc']
                figPlot = plotViolinMCC(figPlot, modelMCCs, modelsNames[i], protInfoList, modelColors[i])
            return figPlot
        
        def plotCombinedScatterAvgMCC(numModels, protInfoList):
            def createScatterAvgMCCSubplots():
                figPlot = make_subplots(rows=1, cols=1,
                                        subplot_titles=['Average MCC Scatters for @{}@'.format(datasetName),
                                                        ],
                                        ) 
                figPlot.update_layout(dict(
                                   autosize = False,
                                  ))
                return figPlot
            
            def plotScatterAvgMCC(figPlot, avgMCCs, stdMCCs, ensModelMCCs, protInfoList):
                trace = go.Scatter(x=avgMCCs,y=stdMCCs,hoverinfo='x+y+text',text=protInfoList,mode='markers',
                                   showlegend=True,name='other-models', marker_size=10)
                figPlot.add_trace(trace,1,1)
                trace = go.Scatter(x=ensModelMCCs,y=stdMCCs,hoverinfo='x+y+text',text=protInfoList,mode='markers',
                                   showlegend=True,name='ensnet-model')
                figPlot.add_trace(trace,1,1)
                figPlot.update_xaxes(patch={'title': 'Ensnet MCC/Avg MCCs per prot across other-models'},row=1,col=1)
                figPlot.update_yaxes(patch={'title': 'Std of MCCs per prot across other-models'},row=1,col=1)
                return figPlot
            
            figPlot = createScatterAvgMCCSubplots()
            modelsMCCs = {}
            for i in range(numModels-1):
                modelMetricsDF = metricsPerModel[i]
                modelMCCs = modelMetricsDF['prot_mcc']
                modelsMCCs[modelsNames[i]] = modelMCCs
            tmpDF = pd.DataFrame(modelsMCCs)
            avgMCCs = tmpDF.mean(axis=1)
            stdMCCs = tmpDF.std(axis=1)
            
            ensModelMetricsDF = metricsPerModel[-1]    #this must be the ensnet metrics.
            ensModelMCCs = ensModelMetricsDF['prot_mcc']
            
            figPlot = plotScatterAvgMCC(figPlot, avgMCCs, stdMCCs, ensModelMCCs, protInfoList)
            return figPlot
        
        def plotCombinedScatterAvgMCC_back(numModels, protInfoList):
            def createScatterAvgMCCSubplots():
                figPlot = make_subplots(rows=1, cols=1,
                                        subplot_titles=['Average MCC Scatters for @{}@'.format(datasetName),
                                                        ],
                                        ) 
                figPlot.update_layout(dict(
                                   autosize = False,
                                  ))
                return figPlot
            
            def plotScatterAvgMCC(figPlot, avgMCCs, stdMCCs, protInfoList):
                trace = go.Scatter(x=avgMCCs,y=stdMCCs,hoverinfo='x+y+text',text=protInfoList,mode='markers',
                                   showlegend=False,)
                figPlot.add_trace(trace,1,1)
                figPlot.update_xaxes(patch={'title': 'Avg MCCs per prot across methods'},row=1,col=1)
                figPlot.update_yaxes(patch={'title': 'Std of MCCs per prot across methods'},row=1,col=1)
                return figPlot
            
            figPlot = createScatterAvgMCCSubplots()
            modelsMCCs = {}
            for i in range(numModels):
                modelMetricsDF = metricsPerModel[i]
                modelMCCs = modelMetricsDF['prot_mcc']
                modelsMCCs[modelsNames[i]] = modelMCCs
            tmpDF = pd.DataFrame(modelsMCCs)
            avgMCCs = tmpDF.mean(axis=1)
            stdMCCs = tmpDF.std(axis=1)
            #normalize y-axis.
            #stdMCCs = stdMCCs / avgMCCs
            
            figPlot = plotScatterAvgMCC(figPlot, avgMCCs, stdMCCs, protInfoList)
            return figPlot
            
        modelColors = plotly.colors.DEFAULT_PLOTLY_COLORS
        numModels = len(modelsNames)
        #we assume that the order of proteins in all model metric files are the same.
        modelMetricsDF = metricsPerModel[0]
        protInfoList = getProtInfoList(modelMetricsDF['prot_id'], modelMetricsDF['prot_len'])
        if plotType == PPIPredPlotCls.VIOLIN_MCC:
            figPlot = plotCombinedViolinMCC(numModels, protInfoList, modelColors)
        elif plotType == PPIPredPlotCls.SCATTER_AVG_MCC:
            figPlot = plotCombinedScatterAvgMCC(numModels, protInfoList)

        return figPlot 
    
class PPIPredPlotCls(object):    
    PRED_COLUMNS = [
                    'prot_id', 
                    'prot_seq', 
                    'y_trues',
                    'y_preds',
                    ]
    
    MERICS_PER_PROT_COLUMNS = [
                    'prot_id', 
                    'prot_len', 
                    'prot_auc', 
                    'prot_ap', 
                    'prot_mcc', 
                    'prot_acc',
                    'prot_pres', 
                    'prot_spec',
                    'prot_sens',
                    'prot_f1',
                    'prot_conf',
                    'cutoff',
                    ]
    
    #Comb plot types
    VIOLIN_MCC = 0
    SCATTER_AVG_MCC = 1
    
    @classmethod 
    def plotProtPreds(cls, predFile, protIds, protSeqs, y_trues, y_preds, datasetName, algorithm):
        def consZvals(protAA):
            AA_Z_DICT = {'A': -0.3454862610776721, 'C': 0.5412500750581398, 'E': 0.2879815393501811, 'D': 0.20388484884574684, 'G': -1.9186298445869447,
                       'F': 2.054314552070933, 'I': 0.17358106139202134, 'H': -0.07884995328923745, 'K': 1.5687285345188036, 'M': 1.0345668036617337, 
                       'L': 1.6612423203053528, 'N': -0.6151910153242968, 'Q': -0.8193308968126504, 'P': -0.7314244102300784, 'S': -1.0690953307421305,
                       'R': -1.3280467722144331, 'T': 0.13400888426922816, 'W': -1.324779568636352, 'V': -0.24501842644784702, 'Y': -1.4055111044293376,
                       'X': -1.481891491787243}
            zvals = [AA_Z_DICT.get(protAA[i]) for i in range(len(protAA))]
            return zvals

        def calcAbsoluteHeight(numProts):
            barHeight = 205 #188
            protHeight = 10
            absHeight = (barHeight * 2) + (protHeight * numProts)
            return absHeight
        
        def calcRelativeHeights(numProts, rowNum):
            barRelHeight = 0.484 #0.474
            protRelHeight = 0.046 #0.056
            totalProtRelHeight = protRelHeight * numProts
            barRelHeight = abs(barRelHeight - (totalProtRelHeight / 2))
            #relHeights = [barRelHeight, barRelHeight]
            relHeights = []
            for i in range(rowNum-1):
                relHeights.append(barRelHeight)
            for i in range(numProts):
                relHeights.append(protRelHeight)
            return relHeights
                
        def plotYtrue(figPlot, rowNum, protId, prot_y_true):
            trace = go.Bar(name=protId,y=prot_y_true,hoverinfo='x+y',showlegend=False,visible=False)
            figPlot.add_trace(trace, rowNum, 1)
            return figPlot
        
        def plotYpred(figPlot, rowNum, protId, prot_y_pred):
            trace = go.Bar(name=protId,y=prot_y_pred,hoverinfo='x+y',showlegend=False,visible=False)
            figPlot.add_trace(trace, rowNum, 1)
            return figPlot
        
        def plotAASeq(figPlot, row, protId, protAA, protLen):
            protId = '<a href=\"https://www.uniprot.org/uniprot/' + protId + '\">' + protId + '</a>'
            trace = go.Bar(
                        y=[1 for _ in range(protLen)],
                        hoverinfo='x',
                        showlegend=False,
                        text=protAA,
                        textfont_size=AA_FONT_SIZE,
                        textposition='inside',
                        marker_color=consZvals(protAA),
                        orientation='v',
                        opacity=AA_DIS_OPACITY,
                        insidetextanchor='middle',
                        marker_showscale=False,
                        marker_colorscale='Rainbow',
                    )
            figPlot.add_trace(trace, row, 1)
            patchDict = {'tickmode':"array",'tickvals':[0.5],'ticks':"outside",'ticktext':[protId]}
            figPlot.update_yaxes(patch=patchDict, row=row,col=1)
            return figPlot
        
        def createPlaceHolderAASeq(figPlot, rowNum):
            protAA = ['M']
            #return plotAASeq(figPlot, 3, 'p-holder', protAA, 1)
            return plotAASeq(figPlot, rowNum, 'p-holder', protAA, 1)
        
        def createSubplots(numProts, rowNum):
            figPlot = make_subplots(rows=numProts+rowNum, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.001, 
                                    row_heights=calcRelativeHeights(numProts+1, rowNum))
            figPlot = createPlaceHolderAASeq(figPlot, rowNum)
            return figPlot
        
        def updateSubplots(figPlot, numProts, longestProtLen):
            figPlot.update_layout(dict(
                       paper_bgcolor="LightSteelBlue",
                       height=calcAbsoluteHeight(numProts+1),
                       width=AA_WIDTH*longestProtLen,
                       autosize = False,
                       margin_b=10,
                       margin_t=40,
                      ))
            return figPlot
        
        def consFigPlot(starti, numProts, placeHolderRowNum):
            figPlot = createSubplots(numProts, placeHolderRowNum)
            selectedProt = protIds[starti]
            #print('######### numProts: ', numProts)
            #row = 4
            row = placeHolderRowNum + 1
            longestProtLen = 0
            for proti in range(0, numProts):
                protId = protIds[starti+proti]
                protAA = protSeqs[starti+proti]
                protLen = len(protAA)
                if protLen >= longestProtLen:
                    longestProtLen = protLen
                if DatasetParams.USE_USERDS_EVAL == False:
                    prot_y_true = y_trues[starti+proti]
                    figPlot = plotYtrue(figPlot, placeHolderRowNum-2, protId, prot_y_true)
                prot_y_pred = y_preds[starti+proti]
                figPlot = plotYpred(figPlot, placeHolderRowNum-1, protId, prot_y_pred)
                figPlot = plotAASeq(figPlot, row, protId, protAA, protLen)
                row = row + 1
                #print('######### protId: ', protId)
            figPlot = updateSubplots(figPlot, numProts, longestProtLen)
            
            #print(figPlot.data)
            #print(figPlot.layout)
            #patchDict = {'opacity':1.0,'y':figPlot.data[3].y,'text':figPlot.data[3].text,'marker':figPlot.data[3].marker}
            
            patchDict = {'opacity':1.0,'y':figPlot.data[placeHolderRowNum].y,'text':figPlot.data[placeHolderRowNum].text,'marker':figPlot.data[placeHolderRowNum].marker}
            figPlot.data[placeHolderRowNum].y = []
            figPlot.data[placeHolderRowNum].text = []
            figPlot.data[placeHolderRowNum].marker = {}
            if DatasetParams.USE_USERDS_EVAL == True:
                figPlot.layout['yaxis2']['ticktext'] = figPlot.layout['yaxis3']['ticktext']
                figPlot.layout['yaxis3']['ticktext'] = ['']
            else:    
                figPlot.layout['yaxis3']['ticktext'] = figPlot.layout['yaxis4']['ticktext']
                figPlot.layout['yaxis4']['ticktext'] = ['']
            figPlot.update_traces(patch=patchDict, row=placeHolderRowNum,col=1)
            figPlot.update_traces(patch={'visible':True}, selector={'name':selectedProt}, row=1,col=1)
            if DatasetParams.USE_USERDS_EVAL == False:
                figPlot.update_traces(patch={'visible':True}, selector={'name':selectedProt}, row=2,col=1)   
            return figPlot
        
        # rowNum-true-values=1; rowNum-pred-values=2;rowNum-placeholder=3; if DatasetParams.USE_USERDS_EVAL==False 
        if DatasetParams.USE_USERDS_EVAL == True:
            placeHolderRowNum = 2
        else: 
            placeHolderRowNum = 3
        
        AA_WIDTH = 30
        AA_FONT_SIZE = 8
        AA_DIS_OPACITY = 0.3
        PROTS_PER_PAGE = 10                 
        updateBarsScript = [
            """
            var gd = document.getElementById('{plot_id}');
            handlePlotlyOnclick(gd);
            """
        ] 
        plotlyOnclickScript = \
            """
            <script> 
            var useUserDSEval = {};
            {}
            </script>
            """ 
        onclickScript = \
            """
            //<script>
            function handlePlotlyOnclick(gd) {
                var prevTraceInd;
                if (useUserDSEval == true) {
                    prevTraceInd = 2;
                }
                else {
                    prevTraceInd = 3;
                }
                gd.on('plotly_click',
                    function(eventData) {{ 
                        //console.log(eventData);
                        //console.log(gd.data);
                        //console.log(gd.layout); 
                        //var t0 = performance.now();
                        opacity = eventData.points[0].fullData.opacity;
                        if (opacity != 1) {
                          figData = gd.data;
                          figLayout = gd.layout;
                          curTraceInd = eventData.points[0].fullData.index;
                          if (curTraceInd != 0 && curTraceInd != prevTraceInd) {
                            firstHeatData = figData[0];
                            prevHeatData = figData[prevTraceInd];
                            curHeatData = figData[curTraceInd];
    
                            prevHeatData.y = firstHeatData.y;
                            prevHeatData.text = firstHeatData.text;
                            prevHeatData.marker = firstHeatData.marker;
                            firstHeatData.y = curHeatData.y;
                            firstHeatData.text = curHeatData.text;
                            firstHeatData.marker = curHeatData.marker;
                            curHeatData.y = [];
                            curHeatData.text = [];
                            curHeatData.marker = {};
    
                            firstYref = firstHeatData.yaxis;
                            firstRowNumStr = firstYref.substring(1); 
                            firstYaxisAttr = 'yaxis' + firstRowNumStr;
                            prevYref = prevHeatData.yaxis;
                            prevRowNumStr = prevYref.substring(1); 
                            prevYaxisAttr = 'yaxis' + prevRowNumStr;
                            curYref = curHeatData.yaxis;
                            curRowNumStr = curYref.substring(1); 
                            curYaxisAttr = 'yaxis' + curRowNumStr;
                            figLayout[prevYaxisAttr].ticktext = figLayout[firstYaxisAttr].ticktext;
                            figLayout[firstYaxisAttr].ticktext = figLayout[curYaxisAttr].ticktext;
                            figLayout[curYaxisAttr].ticktext = [''];
    
                            if (useUserDSEval == false) {
                                figData[curTraceInd-2].visible = true;
                                figData[prevTraceInd-2].visible = false;
                            }
                            figData[curTraceInd-1].visible = true;
                            figData[prevTraceInd].visible = true;
                            figData[prevTraceInd-1].visible = false;
                          
                            Plotly.react(gd, figData, figLayout);
                            prevTraceInd = curTraceInd;
                            window.scrollTo(0, 0);
                            //var t1 = performance.now();console.log(t1-t0);
                            //alert(curTraceInd);
                          }
                        }
                        else {
                          //alert('it was Bar-type');
                        }
                    }}
                );
            }
            //</script>
            """
        useUserDSEval = 'true' if DatasetParams.USE_USERDS_EVAL else 'false'
        plotlyOnclickScript = plotlyOnclickScript.format(useUserDSEval, onclickScript)    
        figHtmlDiv = \
            """
            <div class="list-group" {}>
                {}
            </div>
            """
        if DatasetParams.USE_USERDS_EVAL == True:
            figDivs = ''
        else:
            figPlot = PPIPredMericsCls.consFigMetrics(y_trues, y_preds)
            figDiv = figPlot.to_html(full_html=False, include_plotlyjs=False)
            figDivs = figHtmlDiv.format('', figDiv)
        
        totalNumProts = len(protIds) 
        numPages = (totalNumProts // PROTS_PER_PAGE) 
        remProts = totalNumProts % PROTS_PER_PAGE
        if remProts != 0:
            numPages = numPages + 1
        starti = 0
        numProts = PROTS_PER_PAGE
        style = 'style="display:none"'
        for i in range(0,numPages):
            if remProts !=  0 and i == numPages-1:
                numProts = remProts     # the last page can contain less prots than PROTS_PER_PAGE
            figPlot = consFigPlot(starti, numProts, placeHolderRowNum)
            figDiv = figPlot.to_html(full_html=False, include_plotlyjs=False, post_script=updateBarsScript)
            figDivs = figDivs + figHtmlDiv.format(style, figDiv)
            starti = starti + numProts
    
        paginationJS = \
          """
          <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
          <script>
                'use strict';
              function generatePageRange(currentPage, lastPage) {
                  const delta = 3;
                  const range = [];
                  for (let i = Math.max(2, (currentPage - delta)); i <= Math.min((lastPage - 1), (currentPage + delta)); i += 1) {
                      range.push(i);
                  }
                  if ((currentPage - delta) > 2) {
                      range.unshift('...');
                  }
                  if ((currentPage + delta) < (lastPage - 1)) {
                      range.push('...');
                  }
                  range.unshift(1);
                  if (lastPage !== 1) range.push(lastPage);
                  return range;
              }
        
              var numberOfItems = $('#page .list-group').length; // Get total number of the items that should be paginated
              var limitPerPage = 1; // Limit of items per each page
              $('#page .list-group:gt(' + (limitPerPage - 1) + ')').hide(); // Hide all items over page limits (e.g., 5th item, 6th item, etc.)
              var totalPages = Math.round(numberOfItems / limitPerPage); // Get number of pages
        
              function drawPageNav(curPageInt){
                $(".pagination").empty();
                
                $(".pagination").append("<li id='previous-page'><a href='javascript:void(0)' aria-label=Previous><span aria-hidden=true>&laquo;</span></a></li>");
                var range = generatePageRange(curPageInt, totalPages);
                for (var i = 0; i < range.length; i++) {
                  $(".pagination").append("<li class='current-page'><a href='javascript:void(0)'>" + range[i] + "</a></li>"); 
                }    
                var curPageIndex = range.indexOf(curPageInt, 0);
                $(".pagination").append("<li id='next-page'><a href='javascript:void(0)' aria-label=Next><span aria-hidden=true>&raquo;</span></a></li>");
                
                $(".pagination li.current-page").removeClass('active'); 
                $(".pagination li.current-page:eq(" + curPageIndex + ")").addClass('active');
                
                $("#page .list-group").hide();     
                $("#page .list-group:eq(" + (curPageInt-1) + ")").show(); 
                //console.log($(".pagination").contents());
              }
        
              drawPageNav(1);
        
              // Function that displays new items based on page number that was clicked
              $(document).on("click", ".pagination li.current-page", function() {
                if ($(this).hasClass('active')) {
                  return false; 
                }
                var curPage = $(this).text(); 
                var curPageInt = parseInt(curPage);
                if (isNaN(curPageInt)) {
                  return false;
                } 
                drawPageNav(curPageInt);
              });
        
              // Function to navigate to the next page when users click on the next-page id (next page button)
              $(document).on("click", "#next-page", function() {
                var curPage = $(".pagination li.active").text(); 
                var curPageInt = parseInt(curPage);
                if (curPageInt == totalPages) {
                  return false; 
                }
                drawPageNav(curPageInt+1);
              });
        
              // Function to navigate to the previous page when users click on the previous-page id (previous page button)
              $(document).on("click", "#previous-page", function() {
                var curPage = $(".pagination li.active").text(); 
                var curPageInt = parseInt(curPage);
                if (curPageInt == 1) {
                  return false; 
                }
                drawPageNav(curPageInt-1);
              });
          </script>
          """
        # Note that we remove the suffix "-ppi" from the name of the algorithm (ensnet-ppi --> ensnet).
        figTitle = 'Plot of I/NI-AAs (y_true and y_pred) of @{}@ containing {} proteins predicted by @{}@: '.format(datasetName, totalNumProts, algorithm[:-4])  
        template = \
          """
          <html>
            <head>
              <meta charset="utf-8" />
              <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.1/css/bootstrap.min.css">
              <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
              {}
            </head>
            <body>
              <div>
                <div class="container-fluid">    
                    <div><p>   </p></div>
                    <nav aria-label="...">          
                      <b>{}</b>
                      <ul class="pagination" style="margin: 0px !important; vertical-align: middle;">
                        <li id="previous-page"><a href="javascript:void(0)" aria-label=Previous><span aria-hidden=true>&laquo;</span></a></li>
                        <li class='current-page'><a href='javascript:void(0)'> 1 </a></li>
                        <li id="next-page"><a href="javascript:void(0)" aria-label=Next><span aria-hidden=true>&raquo;</span></a></li>
                      </ul>
                    </nav>
                    <div><p>   </p></div> 
                    <div id="page">
                      {}
                    </div>
                </div>
              </div>
              {}
            </body>
          </html>
          """
        head, tail = os.path.split(predFile)
        tail = 'PPLOTS_' + tail
        slash = '/'
        if head == '':
            slash = ''
        figFile = head + slash + tail
        figFile = figFile.replace('.csv', '.html')
        with open(figFile, 'w') as f:
            f.write(template.format(plotlyOnclickScript, figTitle, figDivs, paginationJS))
        print("## Prediction plot generated for: " + figFile)
        
        return figFile
    
    @classmethod
    def plotStackedProtPreds(cls, predFile, y_trues, y_stacked_preds, modelsNames, datasetName):
        head, tail = os.path.split(predFile)
        tail = 'SROCS_' + tail
        slash = '/'
        if head == '':
            slash = ''
        figFile = head + slash + tail
        figFile = figFile.replace('.csv', '.html')
        figPlot = PPIPredMericsCls.consStackedFigMetrics(y_trues, y_stacked_preds, modelsNames, datasetName)
        plotly.io.write_html(figPlot, figFile)
        print("## Combined AUC & PR plots generated for: " + figFile)
        
        return figFile
    
    @classmethod
    def plotCombinedMetricsPerProt(cls, metricsFile, metricsPerModel, modelsNames, datasetName, plotType):
        head, tail = os.path.split(metricsFile)
        if plotType == PPIPredPlotCls.VIOLIN_MCC:
            prefix = 'Comb_VM_'
        elif plotType == PPIPredPlotCls.SCATTER_AVG_MCC:
            prefix = 'Comb_SM_'
            
        tail = prefix + tail
        slash = '/'
        if head == '':
            slash = ''
        figFile = head + slash + tail
        figFile = figFile.replace('.csv', '.html')
        figPlot = PPIPredMericsCls.consCombinedPerProtFigMetrics(metricsPerModel, modelsNames, datasetName, plotType)
        plotly.io.write_html(figPlot, figFile)
        print("## Combined metrics per protein plots generated for: " + figFile)
        
        return figFile
    
    @classmethod 
    def generatePredPlot(cls, predFile, datasetName='my-dataset', algorithm='my-algorithm'):
        FLOAT_TYPE = 'float64'
        dataset = pd.read_csv(predFile)
        protIds = dataset.loc[:, cls.PRED_COLUMNS[0]].values
        protSeqs,y_trues,y_preds = [],[],[]
        numProts = len(protIds)
        for proti in range(0,numProts):
            protAA = dataset.iloc[proti, dataset.columns.get_loc(cls.PRED_COLUMNS[1])].split(',')
            protSeqs.append(protAA)
            prot_y_true = dataset.iloc[proti, dataset.columns.get_loc(cls.PRED_COLUMNS[2])]
            prot_y_true = np.fromstring(prot_y_true, dtype=FLOAT_TYPE, sep=',').tolist()
            y_trues.append(prot_y_true)
            prot_y_pred = dataset.iloc[proti, dataset.columns.get_loc(cls.PRED_COLUMNS[3])]
            prot_y_pred = np.fromstring(prot_y_pred, dtype=FLOAT_TYPE, sep=',').tolist()
            y_preds.append(prot_y_pred)
        
        figFile = cls.plotProtPreds(predFile, protIds, protSeqs, y_trues, y_preds, datasetName, algorithm)
        
        return figFile
    