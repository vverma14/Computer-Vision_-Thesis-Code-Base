#
# script to launch the training of pre-trained cnn.
#

import sys
from utils.launcher_module import start_with_json
import torch.multiprocessing as mp ## Checking for spawn methods

#-----------------------#
# A) Parameters   #
#-----------------------#
# hyper-parameters
#  pre-trained network:
#    . efficientnet-b0, ..., efficientnet-b8
#    . resnet18, resnet34, resnet50, resnet101
#    . densenet121, densenet169, densenet201, densenet161
#    . vgg16
P = {'nbr_epoch' : 8,
     'learning_rate' : 0.0001,
     'decay_lr' : 1.25,
     'min_decay_loss' : .99,
     'batch_size' : 4,
     'nbr_workers' : 4,
     'nbr_classes': 2,
     'weight': [0.08601, 0.91399],
     'class_names': ['Nevus', 'Melanoma'],
     'pct_train_set': .8,
     'shuffle_dataset': True,
     'weight_pct_trainSet': False,
     'folder_result': 'results',
     'model': 'efficientnet-b1', # resnet, densenet, vgg
     'nbr_meta_features': 11,
     'dropout_meta': .5,
     'epoch_unfreeze': 1,         # <1 to unfreeze all the parameters
     'pretrained': True}
#-----------------------#
# B) data-augmentation  #
#-----------------------#
P['transformation_train'] = ("Compose(["
                              "RandomHorizontalFlip(),"
                              "RandomRotation(180),"
                              "RandomResizedCrop(224,scale=(0.5, 1.0)),"
                              #"RandomResizedCrop(224),"
                              "ColorJitter(brightness=.2, contrast=.2, saturation=.2, hue=.2),"
     	                      "ToTensor(),"
                              "Normalize([0.74694,0.58144,0.56228],[0.15022,0.13995,0.15327])])")
#Original tranformation that was applied : Normalize([0.74694,0.58144,0.56228],[0.15022,0.13995,0.15327])])
#"transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])")
#This is the segmentation Normalize(mean=[0.708, 0.582, 0.536],std=[0.0978, 0.113, 0.127])]
P['transformation_test'] = ("Compose(["
			    "Resize(224),"
			    "CenterCrop(224),"
			    "ToTensor(),"
                            "Normalize([0.74694,0.58144,0.56228],[0.15022,0.13995,0.15327])])")
                            ##"Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])")
#-----------------------#
# C) data-set           #
#-----------------------#
# data set 2020
P['csv_file'] = '../../data/data_ISIC_combined_2019_2020/folds_13062020.csv'
P['path_data'] = '../../data/ISIC_2020_jpeg/train/'
#---------------------#
# D) Start training   #
#---------------------#
if __name__ == "__main__":
   mp.set_start_method('spawn')
   start_with_json(sys.argv[1:], P)

# start from a json folder:
#--------------------------
#   python main_train.py --jsonfolder ../results/folder_json_torun
