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
from scipy.interpolate import interp1d

# A) load (hyper) parameters
#-------------------
pathFolderResult = '../results/dropout/Report_2020-02-06_21h58m44_batchnorm_denseNet201/'
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
myModel.eval()     # faster (no training)

# C) prediction network ('proba' melanoma)
#-----------------------------------------
image = Image.open('../results/testing_network/ISIC_0055550_class0.jpg')
myTransform = eval(P['transformation_test'])
image_tensor = myTransform(image)
with torch.no_grad():
    score_class = myModel(image_tensor.unsqueeze(0))
    proba_class = torch.softmax(score_class,dim=1)
    proba_np = proba_class.squeeze(0).numpy() # back in numpy instead of torch
# the value
y_predict = proba_np[0] + proba_np[4] # we combine 'Melanoma' and 'BKL'
print(" --output network: {:04.4f}".format(y_predict))
    
# D) determine the level of danger
#---------------------------------
df_estimate_proba = pd.read_csv(pathFolderResult+'estimation_proba.csv')
interp_estimate_proba = interp1d(df_estimate_proba['x'], df_estimate_proba['p_hat'],kind='linear')
interp_estimate_std = interp1d(df_estimate_proba['x'], df_estimate_proba['p_std'],kind='linear')
y_hat = interp_estimate_proba(y_predict)
std_hat = interp_estimate_std(y_predict)
print(" --estimate proba: {:04.4f}".format(y_hat))
print("  with confidence interval: {:04.4f} < ... < {:04.4f}".format(y_hat-std_hat,y_hat+std_hat))
