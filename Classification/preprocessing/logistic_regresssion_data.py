import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
from sklearn.impute import SimpleImputer

# the data
df = pd.read_csv('../train_ISIC_2020.csv')
X = pd.concat([pd.get_dummies(df['sex']),
               pd.get_dummies(df['anatom_site_general_challenge']).iloc[:,0:5],
               df['age_approx']],axis=1,ignore_index=True)
X = pd.concat([pd.get_dummies(df['sex']),
               pd.get_dummies(df['anatom_site_general_challenge']),
               df['age_approx']],axis=1)
y = df['target']
# remove missing data with the mean
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X_bis = imp.transform(X)
# fitting
theModel = sklearn.linear_model.LogisticRegressionCV()
theModel.fit(X_bis,y)
# print theModel.coef_


## tp = np.array([np.ones(33126),X_bis[:,8]])
## theModel.fit(tp.T,y)


import statsmodels.api as sm
model2 = sm.Logit(y,X_bis)
result = model2.fit(method='newton')
