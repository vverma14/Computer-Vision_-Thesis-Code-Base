
# -*- coding: utf-8 -*-u
# ML
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import scipy
# classic
import numpy as np
import pandas as pd
import itertools
import matplotlib
matplotlib.use('agg')      # need it on the server (no GUI)
import matplotlib.pyplot as plt
import os

#from IPython.core.debugger import Tracer

def stat_cm(cm):
    ''' Estimate the accuracy (i.e. (TP+TN)/(P+N)) and the 'balanced' accuracy
        (i.e. 1/2(TP/P + TN/N)) of a confusion matrix '''
    accuracy = cm.diagonal().sum() / cm.sum()
    sum_row_noZero = np.sum(cm,axis=1) + (np.sum(cm,axis=1)==0)
    bal_acc = np.mean( cm.diagonal()/sum_row_noZero )
    return accuracy,bal_acc

def plot_stat_training(df,folder_name):
    ''' statistics over epochs '''
    # init
    nbEpochs = len(df) - 1
    # plot
    plt.figure(1);plt.clf()
    plt.ioff()
    plt.plot(df['epoch'],df['loss'],'o-',color='green')
    plt.plot(df['epoch'],df['accuracy_train'],'o-',color='red')
    plt.plot(df['epoch'],df['accuracy_test'],'o-',color='orange',linestyle='dashed')
    plt.plot(df['epoch'],df['bal_acc_train'],'o-',color='blue')
    plt.plot(df['epoch'],df['bal_acc_test'],'o-',color='teal',linestyle='dashed')
    #plt.plot(df['epoch'],df['lr']/df['lr'][0],'o-',color='brown')
    plt.grid(b=True, which='major')
    plt.xlabel(r'epoch')
    plt.ylabel(r'loss')
    plt.legend([r'loss',r'accuracy',r'accuracy (test)',
                r'balanced acc',r'balanced acc (test)'],loc=0)
    plt.axis([-.5, nbEpochs+.5, 0, 1])
    plt.draw()
    plt.savefig(folder_name+'/stat_epochs.pdf')
    plt.close()

def plot_confusion_matrix(cm, classes, folder_name, name_ext, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        sum_row = cm.sum(axis=1)[:, np.newaxis]
        sum_row += (sum_row==0)  # in case of division by zero
        cm = cm.astype('float') / sum_row
    # plot
    plt.figure(1);plt.clf()
    plt.ioff()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    # deco
    plt.colorbar()
    tick_marks = np.arange(len(classes))    
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion matrix')
    # bug matplotlib 3.1.1
    #ax = plt.gca()
    #bottom, top = ax.get_ylim()
    #ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.draw()
    plt.tight_layout()
    plt.savefig(folder_name+'/cm_'+name_ext+'.pdf')
    plt.close()

def plot_ROC(df,folderName):
    ''' Estimate ROC curve, false-pos/neg rate
          . df: dataframe with columns
                     'proba class 0',...,'proba class k-1',true_label
    '''
    # A) AUC
    nbr_classes = len(df.columns)-1
    y_true = df['true_label'].to_numpy(dtype=int)
    y_probas = df.iloc[:,range(0,nbr_classes)].to_numpy(dtype=float)
    # plot
    plt.figure(1);plt.clf()
    plt.title('Area Under the Curve (AUC)')
    for k in range(nbr_classes):
        fpr, tpr, thresholds  = roc_curve(y_true==k, y_probas[:,k])
        myAuc = auc(fpr, tpr)
        plt.plot(fpr, tpr,label = '{0} (area = {1:0.2f})'.format(df.columns[k], myAuc))
    # decoration
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.grid(b=True, which='major')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.0])
    plt.xlabel('False-Positive Rate')
    plt.ylabel('True-Positive Rate')
    plt.draw()
    plt.savefig(folderName+'/AUC_test.pdf')
    plt.close()
    # B) miss rate Melanoma
    # init
    y_true_mel = (df['true_label'] == 1)
    y_proba_mel = df['Nevus']
    fpr, tpr, thresholds  = roc_curve(y_true_mel, y_proba_mel)
    df_ROC = pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds})
    df_ROC.to_csv(folderName+'/ROC_stat.csv',index=False)
    precision, recall, thresholds = precision_recall_curve(y_true_mel, y_proba_mel)
    df_Pre = pd.DataFrame({'precision':precision[:-1], 'recall':recall[:-1], 'thresholds':thresholds})
    df_Pre.to_csv(folderName+'/Precision_stat.csv',index=False)
    # estimate proba
    x = y_proba_mel.to_numpy(dtype='float')
    y = y_true_mel.values + 0
    xMin = np.min(x)-.05
    xMax = np.max(x)+.05
    dx = (xMax-xMin)/2000
    xInt = np.arange(xMin,xMax,dx)
    p_hat,p_std = estimate_p_bernouilli_weight(x,y,xInt,50*dx)
    plt.figure(2);plt.clf()
    plt.plot(y_proba_mel,y_true_mel,'o',label="test sets")
    plt.plot(xInt,p_hat,color='brown',label="estimate proba MELANOME (test set)")
    plt.plot(xInt,p_hat+p_std,'-.',color='blue',label="confidence interval proba")
    plt.plot(xInt,p_hat-p_std,'-.',color='blue')
    plt.grid()
    plt.legend()
    plt.xlabel('output network (y_proba)')
    plt.ylabel('true label')
    plt.draw()
    plt.savefig(folderName+'/estimation_proba.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
    # saving
    df_estimate = pd.DataFrame({'x':xInt, 'p_hat':p_hat, 'p_std':p_std})
    df_estimate.to_csv(folderName+'/estimation_proba.csv',index=False)

def estimate_p_bernouilli_weight(x,y,xInt,sigma):
    '''Estimate B(p) conditioning to x'''
    n = len(xInt)
    p_hat = np.zeros(n)
    p_std = np.zeros(n)
    for i,xi in enumerate(xInt):
        w_i = np.exp(-(x-xi)**2/sigma**2) / np.exp(0.0)
        n_i = w_i.sum()
        p_hat[i] = np.sum(y*w_i)/n_i
        p_std[i] = 1.96*np.sqrt(p_hat[i]*(1-p_hat[i])/n_i)
    return p_hat,p_std
