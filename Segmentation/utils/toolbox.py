# Pytorch libraries
import torch
import torch.nn.functional as F
# classic libraries
import pandas as pd
import matplotlib
matplotlib.use('agg')      # need it on the server (no GUI)
import matplotlib.pyplot as plt

def plot_stat_training(df,folder_name):
    ''' statistics over epochs '''
    # init
    nbEpochs = len(df) - 1
    # plot
    plt.figure(1);plt.clf()
    plt.ioff()
    plt.plot(df['epoch'],df['loss_train'],'o-',color='blue',linestyle='dashed',label='loss train')
    plt.plot(df['epoch'],df['loss_test'],'o-',color='teal',label='loss test')
    plt.plot(df['epoch'],df['jaccard_train'],'o-',color='red',linestyle='dashed',label='Jaccard train')
    plt.plot(df['epoch'],df['jaccard_test'],'o-',color='orange',label='Jaccard test')
    plt.plot(df['epoch'],df['lr']/df['lr'][0],'o-',color='brown',label=r'lr ($\times $'+str(df['lr'][0])+')')
    plt.grid(b=True, which='major')
    plt.xlabel(r'epoch')
    plt.ylabel(r'loss')
    plt.legend(loc=0)
    plt.axis([-.5, nbEpochs+.5, 0, 1])
    plt.draw()
    plt.savefig(folder_name+'/stat_epochs.pdf')
    plt.close()


def compute_jaccard(S,Y):
    '''
    Calcul du Jaccard index entre deux tensors x,y.
      . x de taille: minibatch x 2 (class) x Height x Width
      . y de taille: minibatch x 1 x Height x Width
    '''
    S_bool = torch.ge(S[:,1,:,:],S[:,0,:,:])
    Y_bool = torch.ge(Y[:,0,:,:],1) # >0  Danger!  torch.geg(0,.5) = 1 ! since 0.5->0 for Int
    minibatch = S.shape[0]
    jaccard_total = 0.0
    
    for i in range(minibatch):
        num = (torch.min(S_bool[i,],Y_bool[i,])).sum().item()
        den = (torch.max(S_bool[i,],Y_bool[i,])).sum().item()
        # if (den>0):             # c'est 'unfair' quand y_score=0...
        #     jaccard_total +=  num/den
        jaccard_total +=  num/(den+.000000001)

    return jaccard_total/minibatch

    


    
    
