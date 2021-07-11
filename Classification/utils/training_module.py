# -*- coding: utf-8 -*-
# torch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import efficientnet_pytorch
from torchvision import datasets, models, transforms
from torchvision.transforms import *
# classic libraries
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import time
import json,datetime                     # to save dictionary
import sys                               # to stop program if needed
# perso
from utils.loaders_module_4_channel import TrainingLoader   ### CHANGE
from utils.toolbox import *

#from IPython.core.debugger import Tracer

# source:
#   https://discuss.pytorch.org/t/how-to-delete-layer-in-pretrained-model/17648/18
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

    
class Model_cnn_meta(nn.Module):
    # load pre-trained network:
    #    . efficientnet-b0,...,efficientnet-b8
    #    . resnet18, resnet34, resnet50, resnet101
    #    . densenet121, densenet169, densenet201, densenet161
    #    . vgg16
    def __init__(self,P):
        super(Model_cnn_meta, self).__init__()
        # 1) define the cnn part
        #-----------------------
        if ('efficient' in P['model']):
            # efficientnet
            if (P['pretrained']):
                self.cnn = efficientnet_pytorch.EfficientNet.from_pretrained(P['model'])
                ### MADE CHANGES _ ADDING 4TH CHANNEL -- REMOVE IF WOKRING ONLY WITH 3 CHANNELS
                weights = self.cnn._conv_stem.weight.clone()
                self.cnn._conv_stem = nn.Conv2d(4, 32, kernel_size=(3,3),padding= (1,1) ,stride=(2,2), bias=False)
                self.cnn._conv_stem.weight[:, :3] = weights
                self.cnn._conv_stem.weight[:, 3] = self.cnn._conv_stem.weight[:, 0]  ## This doesn't need
                custom_weight = torch.nn.init.xavier_uniform_(torch.zeros([32,3,3]))
                self.cnn._conv_stem.weight[:, 3] = custom_weight
                W = self.cnn._conv_stem.weight.clone().detach()
                W = W.requires_grad_(True)
                self.cnn._conv_stem.weight = nn.Parameter(W)
                # ENDS HERE
            else:
                self.cnn = efficientnet_pytorch.EfficientNet.from_name(P['model'])
            # change last layer
            self.cnn_nbr_features = self.cnn._fc.in_features 
            self.cnn._fc = Identity()
        else:
            # torchvision model
            self.cnn = getattr(torchvision.models,P['model'])(pretrained=P['pretrained'])
            # change last layer
            if ('resnet' in P['model']):
                self.cnn_nbr_features = myModel.fc.in_features
                self.cnn.fc = Identity()
            elif ('vgg' in P['model']):
                self.cnn_nbr_features = 2048
                self.cnn.classifier[6] = Identity()
            elif ('densenet' in P['model']):
                self.cnn_nbr_features = myModel.classifier.in_features # 2208
                self.cnn.classifier = nn.Sequential(nn.BatchNorm1d(num_ftrs),
                                                   Identity())
            else:
                print(' Cannot change the last layer of the network')
                sys.exit()
        # 2) define the meta part
        #------------------------
        self.meta = nn.Sequential(nn.Linear(P['nbr_meta_features'],P['nbr_meta_features']),
                                  nn.BatchNorm1d(P['nbr_meta_features']),
                                  nn.ReLU(),
                                  nn.Dropout(p=P['dropout_meta']))
        # 3) combining layer
        self.fc_combined = nn.Linear(in_features=self.cnn_nbr_features+P['nbr_meta_features'], out_features=P['nbr_classes'], bias=True)
        
    def forward(self, image, meta_data):
        features_cnn = self.cnn(image)
        features_meta = self.meta(meta_data)
        #print(features_cnn.shape)
        #print(features_meta.shape)
        score = self.fc_combined( torch.cat((features_cnn, features_meta), dim=1) )

        return score


def trainModel(myModel,P):
        #-------------------------------------#
        #          A) Loader+network          #
        #-------------------------------------#
        # A.0) GPU?
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        myModel.to(device)
        # A.1) Loader data training
        dataSet = TrainingLoader(P['csv_file'],P['path_data'],P['nbr_classes'],
                                 eval(P['transformation_train']),eval(P['transformation_test']),
                                 P['pct_train_set'],P['shuffle_dataset'])
        # transform the 'string' into a transform object
        dataLoader_train = torch.utils.data.DataLoader(dataSet,
                                                       batch_size=P['batch_size'],
                                                       num_workers=P['nbr_workers'],
                                                       sampler=dataSet.train_sampler)
        dataLoader_test = torch.utils.data.DataLoader(dataSet,
                                                      batch_size=P['batch_size'],
                                                      num_workers=P['nbr_workers'],
                                                      sampler=dataSet.test_sampler)
        nbr_miniBatch_train = len(dataLoader_train)
        # A.2) loss
        param_groups = [
            {'params':myModel.cnn.parameters(),'lr':.0001},
            {'params':myModel.meta.parameters(),'lr':.001},
            {'params':myModel.fc_combined.parameters(),'lr':.001},
        ]
        optimizer = optim.Adam(param_groups)
        weight = torch.FloatTensor(P['weight'])
        criterion = nn.CrossEntropyLoss( weight=weight.to(device) )
        # A.3) other
        df_training = pd.DataFrame(columns=('epoch', 'loss', 'accuracy_train', 'accuracy_test', 'bal_acc_train', 'bal_acc_test'))
        df_proba_test = pd.DataFrame(columns=P['class_names']+['true_label'])
        df_score_test = pd.DataFrame(columns=P['class_names']+['true_label'])
        #-------------------------------------#
        #          B) loop epoch              #
        #-------------------------------------#
        # initialization
        t0 = time.time()
        running_loss_train_old = 1e10
        # start training
        for epoch in range(P['nbr_epoch']):
            # B.0) initialization epoch
            print('Epoch {}/{}'.format(epoch, P['nbr_epoch'] - 1))
            print('-' * 10)
            running_loss_train = 0.0
            running_loss_test = 0.0
            cm_train = np.zeros((P['nbr_classes'],P['nbr_classes']), dtype=int) # confusion matrix
            cm_test  = np.zeros((P['nbr_classes'],P['nbr_classes']), dtype=int)
            # unfreeze
            if (epoch == P.get('epoch_unfreeze',-1)): # default value '-1' in case 'un_preeze_param' not a valid key
                for param in myModel.parameters(): # freeze all layers
                    param.requires_grad = True
            # B.1) training set (gradient descent)
            myModel.train()    # for drop-out and batch-norm
            for data in dataLoader_train:
                # get inputs
                inputs, features, labels = data
                # gradient descent
                optimizer.zero_grad()
                outputs = myModel(inputs.to(device),features.to(device))
                _, preds = torch.max(outputs.data, 1) # to estimate how many are correct
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()
                # statistics
                running_loss_train += loss.item()
                cm_train += confusion_matrix(labels.cpu(),preds.cpu(),labels=range(P['nbr_classes']))
            # B.2) test set
            myModel.eval()     # for drop-out and batch-norm
            with torch.no_grad():
                for data in dataLoader_test:
                    inputs, features, labels = data
                    score = myModel(inputs.to(device),features.to(device))
                    _, preds = torch.max(score.data, 1) # to estimate how many are correct
                    # statistics
                    cm_test += confusion_matrix(labels.cpu(),preds.cpu(),labels=range(P['nbr_classes']))
                    # save proba for the last epoch
                    if (epoch == (P['nbr_epoch']-1)):
                        proba = F.softmax(score,dim=1)
                        n_mini,_ = proba.shape
                        for j in range(n_mini):
                            n_ = len(df_proba_test)
                            df_proba_test.loc[n_,0:P['nbr_classes']] = proba[j,:].cpu().numpy()
                            df_proba_test.loc[n_,'true_label'] = int(labels[j])
                            df_score_test.loc[n_,0:P['nbr_classes']] = score[j,:].cpu().numpy()
                            df_score_test.loc[n_,'true_label'] = int(labels[j])
            # B.3) statistic
            epoch_loss_train = running_loss_train / nbr_miniBatch_train
            accuracy_train,bal_acc_train = stat_cm(cm_train)
            accuracy_test,bal_acc_test = stat_cm(cm_test)
            print('  Loss: {:.4f} '.format(epoch_loss_train))
            print('    Accuracy (train, test): {:.4f},  {:.4f}'.format(accuracy_train, accuracy_test))
            print('    Bal_Acc  (train, test): {:.4f},  {:.4f}'.format(bal_acc_train, bal_acc_test))
            df_training.loc[epoch] = [epoch, epoch_loss_train, accuracy_train, accuracy_test, bal_acc_train, bal_acc_test]
            # B.4) update rate
            # if (running_loss_train > P['min_decay_loss']*running_loss_train_old):
            #     state_dict = optimizer.state_dict()
            #     for param_group in state_dict['param_groups']:
            #         param_group['lr'] /= P['decay_lr']
            #         lr_inside_loop = param_group['lr']
            #     optimizer.load_state_dict(state_dict)
            # the running losses
            running_loss_train_old = running_loss_train
            #?? torch.cuda.empty_cache()
            # one epoch over
        # training over
        #-------------------------------------#
        #          C) save and inspection     #
        #-------------------------------------#
        # C.1) saving network and stat training
        #--------------------------------------
        # C.1a) print time
        time_elapsed = time.time() - t0
        P['time_training'] = '{:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60)
        print('Training complete in '+P['time_training'])
        # C.1b) save network
        str_time = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_').replace(':', 'h', 1).replace(':', 'm', 1)
        os.makedirs(P['folder_result']+'/Report_'+str_time)
        torch.save(myModel.state_dict(), P['folder_result']+'/Report_'+str_time+'/myNetwork.pth')
        P['finish_training'] = str_time
        # C.1c) save parameters
        with open(P['folder_result']+'/Report_'+str_time+'/parameters.json','w') as jsonFile:
             json.dump(P,jsonFile,indent=4)
        # C.1d) save stat training (and a plot)
        df_training.to_csv(P['folder_result']+'/Report_'+str_time+'/stat_epochs.csv',index=False)
        np.save(P['folder_result']+'/Report_'+str_time+'/cm_test_last.npy',cm_test)
        plot_stat_training(df_training,P['folder_result']+'/Report_'+str_time)
        # C.2) inspection classifier
        #---------------------------
        # C.2a) plot confusion matrix
        plot_confusion_matrix(cm_train, P['class_names'],
                              P['folder_result']+'/Report_'+str_time,'train')
        plot_confusion_matrix(cm_train, P['class_names'],
                              P['folder_result']+'/Report_'+str_time,'train_norm',normalize=True)
        plot_confusion_matrix(cm_test, P['class_names'],
                              P['folder_result']+'/Report_'+str_time,'test')
        plot_confusion_matrix(cm_test, P['class_names'],
                              P['folder_result']+'/Report_'+str_time,'test_norm',normalize=True)
        # C.2b) advanced stat (ROC, AUC, False-Pos/Neg rates)
        os.makedirs(P['folder_result']+'/Report_'+str_time+'/df_proba')
        plot_ROC(df_proba_test,P['folder_result']+'/Report_'+str_time+'/df_proba')
        df_proba_test.to_csv(P['folder_result']+'/Report_'+str_time+'/df_proba/df_proba_test.csv',index=False)
        os.makedirs(P['folder_result']+'/Report_'+str_time+'/df_score')
        plot_ROC(df_score_test,P['folder_result']+'/Report_'+str_time+'/df_score')
        df_score_test.to_csv(P['folder_result']+'/Report_'+str_time+'/df_score/df_score_test.csv',index=False)
