import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.linear_model import LogisticRegression

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


# initialization
df = pd.read_csv('../../results/new_weight/Report_2020-02-17_20h10m56_best/df_score/df_score_test.csv')
y_true_mel = (df['true_label'] == 0) | (df['true_label'] == 4) # Melanoma or BKL
y_proba_mel = (df['MEL'] + df['BKL'])
x = y_proba_mel.to_numpy(dtype='float')
y = y_true_mel.values + 0
# estimate seb
xMin = np.min(x)-.05
xMax = np.max(x)+.05
dx = (xMax-xMin)/2000
xInt = np.arange(xMin,xMax,dx)
p_hat,p_std = estimate_p_bernouilli_weight(x,y,xInt,50*dx)
# estimate  RadiusNeighbor
neigh = RadiusNeighborsRegressor(radius=3.0,weights='uniform') # weights='uniform'
neigh.fit(x.reshape(-1,1), y)
p_hat2 = neigh.predict(xInt.reshape(-1,1))
# estimate logistic
myLR = LogisticRegression(random_state=0).fit(x.reshape(-1,1), y)
predict_bool = myLR.predict(xInt.reshape(-1,1))
beta0,beta1 = myLR.intercept_[0],myLR.coef_[0,0]
p_hat3 = myLR.predict_proba(xInt.reshape(-1,1))[:,1]
p_hat3_bis = 1/(1+np.exp(-beta0 - beta1*xInt))

# plot
plt.figure(1);plt.clf()
plt.plot(y_proba_mel,y_true_mel,'o',label="test sets")
plt.plot(xInt,p_hat,color='brown',label="estimate proba MELANOME (test set)")
plt.plot(xInt,p_hat+p_std,'-.',color='blue',label="confidence interval proba")
plt.plot(xInt,p_hat-p_std,'-.',color='blue')
plt.plot(xInt,p_hat2,color='green')
plt.plot(xInt,p_hat3,color='orange')
plt.plot(xInt,p_hat3_bis,color='teal')
plt.grid()
plt.legend()
plt.xlabel('output network (y_proba)')
plt.ylabel('true label')
plt.draw()
