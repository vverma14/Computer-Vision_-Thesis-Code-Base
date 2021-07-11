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

def estimate_p_bernouilli(x,y,xInt,dx):
    '''Estimate B(p) conditioning to x'''
    n = len(xInt)
    p_hat = np.zeros(n)
    p_std = np.zeros(n)
    for i,xi in enumerate(xInt):
        l_i = ((xi-dx)<x)*(x<(xi+dx))
        n_i = len(l_i)
        p_hat[i] = y[l_i].mean()
        p_std[i] = 1.96*np.sqrt(p_hat[i]*(1-p_hat[i])/n_i)
    return p_hat,p_std

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


# init
pathFolderResult = '../../results/dropout/Report_2020-02-06_21h58m44_batchnorm_denseNet201/'
df_score_test = pd.read_csv(pathFolderResult+'/df_score_test.csv')
df_proba_test = pd.read_csv(pathFolderResult+'/df_proba_test.csv')
y_true = (df_proba_test['true_label'] == 0) | (df_proba_test['true_label'] == 4) # Melanoma or BKL
y_proba = (df_proba_test['MEL'] + df_proba_test['BKL'])
y_score = (df_score_test['MEL'] + df_score_test['BKL'])
# local mean proba
xInt = np.arange(.005,1,.005)
x = y_proba.values
y = y_true.values + 0
p_hat,p_std = estimate_p_bernouilli(x,y,xInt,.05)
p_hat2,p_std2 = estimate_p_bernouilli_weight(x,y,xInt,.1)
# plot
plt.figure(1);plt.clf()
plt.plot(y_proba,y_true,'o',label="test sets")
# plt.plot(xInt,p_hat,color='green',label="average proba")
# plt.plot(xInt,p_hat+p_std,'-.',color='blue',label="standard deviation")
# plt.plot(xInt,p_hat-p_std,'-.',color='blue')
plt.plot(xInt,p_hat2,color='brown',label="estimation proba MELANOME (test set)")
plt.plot(xInt,p_hat2+p_std2,'-.',color='blue',label="interval confiance proba")
plt.plot(xInt,p_hat2-p_std2,'-.',color='blue')
plt.grid()
plt.legend()
plt.xlabel('output of the network (y_proba)')
plt.ylabel('true label')
plt.draw()


# local mean score
xInt = np.arange(-20,30,.05)
x = y_score.values
y = y_true.values + 0
p_hat,p_std = estimate_p_bernouilli(x,y,xInt,.05)
p_hat2,p_std2 = estimate_p_bernouilli_weight(x,y,xInt,1)
# plot
plt.figure(2);plt.clf()
plt.plot(y_score,y_true,'o',label="test sets")
# plt.plot(xInt,p_hat,color='green',label="average proba")
# plt.plot(xInt,p_hat+p_std,'-.',color='blue',label="standard deviation")
# plt.plot(xInt,p_hat-p_std,'-.',color='blue')
plt.plot(xInt,p_hat2,color='brown',label="estimation proba MELANOME (test set)")
plt.plot(xInt,p_hat2+p_std2,'-.',color='blue',label="interval confiance proba")
plt.plot(xInt,p_hat2-p_std2,'-.',color='blue')
plt.grid()
plt.legend()
plt.xlabel('output of the network (y_score)')
plt.ylabel('true label')
plt.draw()





sys.exit("Error message")



#-------------------------------------#
#-----          Extra            -----#
#-------------------------------------#

# minimize
from scipy.optimize import minimize
x = y_proba.values
y = y_true.values + 0
def myLoss(alpha):
    weight = y + 10
    return np.mean( weight*(y-x**alpha)**2 )
res = minimize(myLoss, [1], method='Nelder-Mead')
theAlpha = res.x[0]
# def myLoss_noWeight(alpha):
#     return np.mean( (y-x**alpha)**2 )
# res_noWeight = minimize(myLoss_noWeight, [1], method='Nelder-Mead')
# def myLoss_intercept(alpha):
#     weight = y + 10
#     return np.mean( weight*(y-(alpha[0]+x**alpha[1]))**2 )
# res2 = minimize(myLoss_intercept, [0,1], method='Nelder-Mead')
## loess with python
## import statsmodels.api as sm
## lowess = sm.nonparametric.lowess
## z = lowess(y, x)
# with R
##import subprocess
##subprocess.call("exit 1", shell=True)
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
xInt = np.arange(.005,1,.005)
x_inR = robjects.FloatVector(x)
y_inR = robjects.FloatVector(y)
xInt_inR = robjects.FloatVector(xInt)
stats_inR = importr('stats')
#msir_inR = importr('msir')
robjects.globalenv["x"] = x_inR
robjects.globalenv["y"] = y_inR
model_inR = stats_inR.loess("y~x")
y_predict = np.array( stats_inR.predict(model_inR,xInt_inR) )
# sd
y_tp_predict = np.array( stats_inR.predict(model_inR,x_inR) )
y_sd_inR = robjects.FloatVector( np.sqrt( (y-y_tp_predict)**2 ) )
robjects.globalenv["y_sd"] = y_sd_inR
model_sd_inR = stats_inR.loess("y_sd~x")
sd_predict = np.array( stats_inR.predict(model_sd_inR,xInt_inR) )

# final plot
plt.figure(1);plt.clf()
plt.plot(y_proba,y_true,'o',label="test sets")
plt.plot(xInt,y_predict,color='red',label="average proba")
plt.plot(xInt,y_predict-sd_predict/5,'-.',color='orange',label="standard deviation")
plt.plot(xInt,y_predict+sd_predict/5,'-.',color='orange')
plt.grid()
plt.legend()
plt.xlabel('output network')
plt.ylabel('true label')
plt.draw()

plt.figure(2);plt.clf()
plt.plot(y_proba,y_true,'o',label="test sets")
plt.plot(xInt,xInt**theAlpha,'purple',label="average proba (function)")
plt.plot(xInt,xInt**theAlpha-sd_predict/5,'-.',color='orange',label="standard deviation")
plt.plot(xInt,xInt**theAlpha+sd_predict/5,'-.',color='orange')
plt.grid()
plt.legend()
plt.xlabel('y network')
plt.ylabel('true label')
plt.draw()
