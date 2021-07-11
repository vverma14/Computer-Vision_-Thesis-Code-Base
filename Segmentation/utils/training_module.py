# Pytorch
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms import * # Compose, RandomHorizontalFlip...
# classic
import datetime, json, os, sys
import numpy as np
import pandas as pd
import time
# perso
from utils.dataset_module import Mole_dataset
from utils.prediction_module import prediction_model
from utils.all_models.unet import UNet
from utils.toolbox import plot_stat_training, compute_jaccard
#from IPython.core.debugger import Tracer

def trainModel(P):
    #---------------------------#
    #        A.1) Dataset       #
    #---------------------------#
    dataSet = Mole_dataset(P['path_images'], P['path_masks'],
                          eval(P['img_transform_train']), eval(P['mask_transform_train']),
                          eval(P['img_transform_test']), eval(P['mask_transform_test']),
                          P['pct_train_set'], P['shuffle_dataset'])
    if (dataSet.__len__()==0):
        print("--- Problem initialization: dataSet empty\n")
        sys.exit(2)
    # data-loader
    dataLoader_train = DataLoader( dataSet, batch_size=P['batch_size'], num_workers=P['num_workers'], sampler=dataSet.train_sampler)
    dataLoader_test = DataLoader( dataSet, batch_size=P['batch_size'], num_workers=P['num_workers'], sampler=dataSet.test_sampler)
    nbr_miniBatch = len(dataLoader_train)
    nbr_miniBatch_test = len(dataLoader_test)
    #---------------------------#
    #        A.2) Model         #
    #---------------------------#
    try:
        myModel = eval(P['model'])(3,2) # nbr channels input, nbr classes
    except KeyError:
        print('Key Error: We do not have this model: ' + P['model'].split('(')[0])
        sys.exit(2)
    # cpu/gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    myModel.to(device)
    #---------------------------#
    #        A.3) Loss          #
    #---------------------------#
    weight = torch.tensor([1-P['weight_mole'],P['weight_mole']])
    criterion = nn.CrossEntropyLoss( weight=weight.to(device) )
    optimizer = torch.optim.Adam(myModel.parameters(), lr=P['lr'])
    #-------------------------------------------------#
    #-------------------------------------------------#
    #                  B) Training                    #
    #-------------------------------------------------#
    #-------------------------------------------------#
    # initialize
    df_training = pd.DataFrame(columns=('epoch', 'lr', 'loss_train', 'loss_test',
                                        'jaccard_train', 'jaccard_test'))
    loss_train_old = 1e10
    lr_inside_loop = P['lr']
    t0 = time.time()
    for epoch in range( P['nbr_epoch'] ):  # loop over the dataset multiple times
        # B.0) a new epoch begins
        #------------------------
        print('-- epoch '+str(epoch))
        # B.1) Backpropagation one epoch
        #-------------------------------
        # B.1.a) train set
        myModel.train()
        loss_train,jaccard_train = 0.0,0.0
        for X,Y in dataLoader_train: # X,Y = next(iter(dataLoader_train))
            optimizer.zero_grad()
            S = myModel(X.to(device))
            loss = criterion(S, Y.squeeze(1).to(device))
            loss.backward()
            optimizer.step()
            # statistics
            jaccard = compute_jaccard( S.data.cpu(), Y.data.cpu() )
            cross_entropy = loss.data.item()
            # print/save
            #print('[%d, %5d] cross: %.3f, Jaccard: %.3f' %(epoch, miniBatch, cross_entropy, jaccard))
            loss_train += cross_entropy
            jaccard_train += jaccard
        # B.1.b) test set
        myModel.eval()
        loss_test,jaccard_test = 0.0,0.0
        with torch.no_grad():
            for X,Y in dataLoader_test:
                S = myModel(X.to(device))
                loss = criterion(S, Y.squeeze(1).to(device))
                jaccard = compute_jaccard( S.data.cpu(), Y.data.cpu() )
                cross_entropy_test = loss.data.item()
                # update
                loss_test += cross_entropy_test
                jaccard_test += jaccard
        # B.2) update
        #------------
        #-- print screen
        loss_train /= nbr_miniBatch
        jaccard_train /= nbr_miniBatch
        loss_test /= nbr_miniBatch_test
        jaccard_test /= nbr_miniBatch_test
        print('summary epoch %d:'%(epoch))
        print('   Loss    (train/test):  %.3f,  %.3f' %(loss_train, loss_test))
        print('   Jaccard (train/test):  %.3f,  %.3f' %(jaccard_train, jaccard_test))
        df_training.loc[epoch] = [epoch, lr_inside_loop, loss_train, loss_test, jaccard_train, jaccard_test]
        #-- update rate
        if (loss_train > P['min_decay_loss']*loss_train_old):
            state_dict = optimizer.state_dict()
            for param_group in state_dict['param_groups']:
                param_group['lr'] /= P['decay_lr']
                lr_inside_loop = param_group['lr']
                print('   lr updated: ',lr_inside_loop)
                optimizer.load_state_dict(state_dict)
        #-- the running losses
        loss_train_old = loss_train
        # end epoch
    # training is over
    #-------------------------------------------------#
    #-------------------------------------#
    #          C) save and plot           #
    #-------------------------------------#
    # C.1) print time
    time_elapsed = time.time() - t0
    P['time_training'] = '{:.0f}m {:.0f}s'.format( time_elapsed // 60, time_elapsed % 60)
    print('Training complete in '+P['time_training'])
    # C.2) save network
    str_time = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_').replace(':', 'h', 1).replace(':', 'm', 1)
    os.makedirs(P['folder_result']+'/Report_'+str_time)
    torch.save(myModel.state_dict(), P['folder_result']+'/Report_'+str_time+'/myNetwork.pth')
    P['finish_training'] = str_time
    # C.3) save parameters
    with open(P['folder_result']+'/Report_'+str_time+'/parameters.json','w') as jsonFile:
        json.dump(P,jsonFile,indent=2)
    # C.4) save stat training (and a plot)
    df_training.to_csv(P['folder_result']+'/Report_'+str_time+'/stat_epochs.csv',index=False)
    plot_stat_training(df_training,P['folder_result']+'/Report_'+str_time)
    # C.5) prediction all mask
    #os.makedirs(P['folder_result']+'/Report_'+str_time+'/all_masks')
    #prediction_model(myModel,P,str_time,dataSet.train_sampler)
    # debug 
    #Tracer()()

