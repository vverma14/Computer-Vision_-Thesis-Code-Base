# torch library
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torch.nn.functional as F
# ML
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import scipy
# classic
import numpy as np
from PIL import Image
import pandas as pd
import json, sys
import matplotlib.pyplot as plt
import scipy
# perso
sys.path.append('../')
from utils.loaders_module import TrainingLoader
from utils.toolbox import plot_confusion_matrix, plot_stat_training

shouldPlotTraining = True
shouldComputeThresholds = True

# A) load (hyper) parameters
#-------------------
pathFolderResult = '../../results/new_weight/Report_2020-02-15_03h43m57/'
with open(pathFolderResult+'parameters.json','r') as string:
    P = json.load(string)
# dirty modification
P['nbr_classes'] = 8

# B) load network
#----------------
myModel = getattr(torchvision.models,P['model'])(pretrained=True)
num_ftrs = myModel.classifier.in_features
myModel.classifier = nn.Sequential(nn.BatchNorm1d(num_ftrs),
                                   nn.Linear(num_ftrs,P['nbr_classes']))
#myModel.classifier = nn.Linear(num_ftrs,P['nbr_classes'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myModel.load_state_dict(torch.load(pathFolderResult+'myNetwork.pth',map_location=device))
myModel.to(device)
myModel.eval()     # faster (no training)

# C) prediction network ('proba' melanoma)
#-----------------------------------------
image = Image.open('../../results/testing_network/ISIC_0055550_class0.jpg')
myTransform = eval(P['transformation_test'])
image_tensor = myTransform(image)
with torch.no_grad():
    score_class = myModel(image_tensor.unsqueeze(0).to(device))
    proba_class = torch.softmax(score_class,dim=1)
    proba_np = proba_class.squeeze(0).cpu().numpy() # back in numpy instead of torch

y_predict = proba_np[0] + proba_np[4] # we add class 'Melanoma' and 'BKL'

# D) classic plot
if (shouldPlotTraining):
    cm_test = np.load(pathFolderResult+'cm_test_last.npy')
    plot_confusion_matrix(cm_test, P['class_names'], pathFolderResult,'test')
    df_training = pd.read_csv(pathFolderResult+'stat_epochs.csv')
    plot_stat_training(df_training,pathFolderResult)


# E) determine the level of danger
#---------------------------------
if (shouldComputeThresholds):
    # D.1) find prediction proba data-set
    # loader
    dataSet = TrainingLoader('../'+P['csv_file'],'../'+P['path_data'],P['nbr_classes'],
                             eval(P['transformation_train']),eval(P['transformation_test']),
                             P['pct_train_set'],P['shuffle_dataset'])
    dataLoader_test = torch.utils.data.DataLoader(dataSet,
                                                  batch_size=P['batch_size'],
                                                  num_workers=P['nbr_workers'],
                                                  sampler=dataSet.test_sampler)
    # save all prediction
    df_score_test = pd.DataFrame(columns=P['class_names']+['true_label'])
    df_proba_test = pd.DataFrame(columns=P['class_names']+['true_label'])
    with torch.no_grad():
        for data in dataLoader_test:
            inputs, labels = data
            score = myModel(inputs.to(device))
            proba = F.softmax(score,dim=1)
            n_mini,_ = proba.shape
            for j in range(n_mini):
                n_ = len(df_score_test)
                df_score_test.loc[n_,0:P['nbr_classes']] = score[j,:].cpu().numpy()
                df_score_test.loc[n_,'true_label'] = int(labels[j])
                df_proba_test.loc[n_,0:P['nbr_classes']] = proba[j,:].cpu().numpy()
                df_proba_test.loc[n_,'true_label'] = int(labels[j])
    # save
    df_score_test.to_csv(pathFolderResult+'/df_score_test.csv',index=False)
    df_proba_test.to_csv(pathFolderResult+'/df_proba_test.csv',index=False)
    # D.2) compute ROC information
    y_true = (df_proba_test['true_label'] == 0) | (df_proba_test['true_label'] == 4) # Melanoma or BKL
    y_proba = (df_proba_test['MEL'] + df_proba_test['BKL'])
    fpr, tpr, thresholds  = roc_curve(y_true, y_proba)
    df_ROC = pd.DataFrame({'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds})
    df_ROC.to_csv(pathFolderResult+'/ROC_stat.csv',index=False)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    df_Pre = pd.DataFrame({'precision':precision[:-1], 'recall':recall[:-1], 'thresholds':thresholds})
    df_Pre.to_csv(pathFolderResult+'/Precision_stat.csv',index=False)
    # D.3) plot
    # miss rate
    plt.figure()
    plt.plot(df_ROC['thresholds'],1-df_ROC['tpr'],color='red',label='miss rate')
    plt.plot(df_ROC['thresholds'],df_ROC['fpr'],color='blue',label='false positive rate')
    plt.legend()
    plt.grid(b=True, which='major')
    plt.xlabel(r'thresholds')
    plt.ylabel(r'miss rate/false positive')
    plt.axis([0, 1, -.02, 1.02])
    plt.draw()
    plt.savefig(pathFolderResult+'/missRate_FPR_test.pdf')
    plt.close()
    # precision
    plt.figure()
    plt.plot(df_Pre['thresholds'],df_Pre['precision'],color='teal',label='precision')
    plt.plot(df_Pre['thresholds'],df_Pre['recall'],color='green',label='recall')
    plt.legend()
    plt.grid(b=True, which='major')
    plt.xlabel(r'thresholds')
    plt.ylabel(r'precision/recall')
    plt.axis([0, 1, -.02, 1.02])
    plt.draw()
    plt.savefig(pathFolderResult+'/precision_recall.pdf')
    plt.close()
else:
    df_Pre = pd.read_csv(pathFolderResult+'/Precision_stat.csv')
# D.3) estimate tpr
interp_precision_h = scipy.interpolate.interp1d(df_Pre['thresholds'],df_Pre['precision'],kind='linear')
interp_precision_h(y_predict)
# D.4) testing
df_proba_test = pd.read_csv(pathFolderResult+'/df_proba_test.csv')
y_true = (df_proba_test['true_label'] == 0) | (df_proba_test['true_label'] == 4) # Melanoma or BKL
y_proba = (df_proba_test['MEL'] + df_proba_test['BKL'])
#y_true = (df_proba_test['true_label'] == 0)
#y_proba = df_proba_test['MEL']
# estimation curves
def estimate_p_bernouilli_weight(x,y,xInt,sigma):
    '''Estimate B(p) conditioning to x'''
    n = len(xInt)
    p_hat = np.zeros(n)
    p_std = np.zeros(n)
    for i,xi in enumerate(xInt):
        w_i = np.exp(-(x-xi)**2/sigma**2) / np.exp(0)
        n_i = w_i.sum()
        p_hat[i] = np.sum(y*w_i)/n_i
        p_std[i] = 1.96*np.sqrt(p_hat[i]*(1-p_hat[i])/n_i)
    return p_hat,p_std
# try it
x = y_proba.values
y = y_true.values + 0
#p_hat,p_std = estimate_p_bernouilli(x,y,xInt,.05)
xInt = np.arange(0,1.005,.005)
p_hat,p_std = estimate_p_bernouilli_weight(x,y,xInt,.1)
# plot
plt.figure(1);plt.clf()
plt.plot(y_proba,y_true,'o',label="test sets")
plt.plot(xInt,p_hat,color='brown',label="estimation proba MELANOME (test set)")
plt.plot(xInt,p_hat+p_std,'-.',color='blue',label="interval confiance proba")
plt.plot(xInt,p_hat-p_std,'-.',color='blue')
plt.grid()
plt.legend()
plt.xlabel('output of the network (y_proba)')
plt.ylabel('true label')
plt.draw()
plt.savefig(pathFolderResult+'/estimation_proba.pdf', transparent=True, bbox_inches='tight', pad_inches=0)
# saving
df_estimate_proba = pd.DataFrame({'x':xInt, 'p_hat':p_hat, 'p_std':p_std})
df_estimate_proba.to_csv(pathFolderResult+'/estimation_proba.csv',index=False)



