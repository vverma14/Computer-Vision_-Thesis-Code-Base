# Pytorch libraries
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
# classic libraries
import numpy as np
import matplotlib
matplotlib.use('agg')      # need it on the server (no GUI)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# perso
from utils.dataset_module import Mole_dataset_simple
#from IPython.core.debugger import Tracer

def prediction_model(myModel,P,str_time,train_sampler):
    # simple dataset
    mole_dataset = Mole_dataset_simple(P['path_images'], P['path_masks'])
    N = len(mole_dataset)
    idx_to_show = np.random.randint(N)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize_image = transforms.Normalize(P['mean_RGB'],P['std_RGB'])
    # go over all the dataset
    with torch.no_grad():
        for k in range(N):
            # get the image
            X,y,filename  = mole_dataset.__getitem__(k)
            _,nH,nW = X.shape
            X_center = normalize_image(X)
            if k in train_sampler:
                str_suffix = "_train"
            else:
                str_suffix = "_test"    
            # cut in two squares
            min_HW = min(nH,nW)
            max_HW = max(nH,nW)
            X_left = X_center[0,0:min_HW,0:min_HW]
            X_right = X_center[0,(nH-min_HW):nH,(nW-min_HW):nW]
            # prediction
            X_left_ready = F.interpolate(X_left.view(1,1,min_HW,min_HW),
                                         size=P['resize'],mode='bilinear')
            proba_mask_left = F.softmax( myModel(X_left_ready.to(device)),dim=1 ).data.cpu()
            X_right_ready = F.interpolate(X_right.view(1,1,min_HW,min_HW),
                                          size=P['resize'],mode='bilinear')
            proba_mask_right = F.softmax( myModel(X_right_ready.to(device)),dim=1 ).data.cpu()
            # interpolate back
            proba_mask_left_big = F.interpolate(proba_mask_left,
                                                size=min_HW,mode='bilinear')
            proba_mask_right_big = F.interpolate(proba_mask_right,
                                                 size=min_HW,mode='bilinear')
            # pad
            pad1 = (0,max_HW-nH,0,max_HW-nW)
            tp_left = F.pad(proba_mask_left_big,pad1)
            pad2 = (max_HW-nH,0,max_HW-nW,0)
            tp_right = F.pad(proba_mask_right_big,pad2)
            # finally...
            proba_mole = torch.max(tp_left[0,1,:,:],tp_right[0,1,:,:])
            # plot
            plt.figure(42);plt.clf()
            plt.imshow(proba_mole>.5)
            plt.draw()
            plt.savefig(P['folder_result']+'/Report_'+str_time+"/all_masks/"+filename+str_suffix+".jpg",bbox_inches='tight')
            # show example
            if (k == idx_to_show):
                strName = P['folder_result']+'/Report_'+str_time+"/example.jpg"
                show_example(X.squeeze(0),y.squeeze(0),proba_mole,strName)
            # debug 
            #Tracer()()
                        

def show_example(X, y, proba, strName):
    # plot
    plt.figure(1);plt.clf()
    plt.subplot(2,2,1); plt.imshow(X,cmap='gray',vmin=0, vmax=1)
    plt.subplot(2,2,2); plt.imshow(y)
    ax_tp = plt.subplot(2,2,3); img1 = ax_tp.imshow(proba)
    divider = make_axes_locatable(ax_tp) # for the colorbar
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(img1, cax=cax1)
    plt.subplot(2,2,4); plt.imshow(proba>.5)
    plt.draw()
    plt.savefig(strName,bbox_inches='tight')

