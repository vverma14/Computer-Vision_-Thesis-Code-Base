"""
   Launcher for the training of a U-Net network to segment mole images.
"""

import sys
from utils.json_launcher_module import start_with_json
from utils.training_module import trainModel


#-----------------------#
# A) Parameters   #
#-----------------------#
# model
P = {'nbr_epoch' : 20,
     'lr' : 0.0001,
     'decay_lr' : 1.25,
     'min_decay_loss' : .99,
     'batch_size' : 1,
     'num_workers' : 8,
     'weight_mole': .7,
     'pct_train_set': .8,
     'shuffle_dataset': True,
     'folder_result': '../results',
     'model': 'UNet',
     'path_images':'../mini_dataset/images_mole/',
     'path_masks':'../mini_dataset/images_mole_mask/',
     'mean_RGB': [0.708, 0.582, 0.536],
     'std_RGB': [0.0978, 0.113, 0.127],
     'resize': 400
}
#-----------------------#
# B) data-augmentation  #
#-----------------------#
P['img_transform_train'] = ("Compose(["
                            "RandomHorizontalFlip(),"
                            "RandomRotation(180),"
                            "RandomResizedCrop("+str(P['resize'])+",scale=(0.5, 1.0)),"
     	                    "ToTensor(),"
                            "Normalize("+str(P['mean_RGB'])+","+str(P['std_RGB'])+")])")
P['mask_transform_train'] = ("Compose(["
                             "RandomHorizontalFlip(),"
                             "RandomRotation(180),"
                             "RandomResizedCrop("+str(P['resize'])+",scale=(0.5, 1.0)),"
     	                     "ToTensor(),"
                             "])")
P['img_transform_test'] = ("Compose(["
			   "Resize("+str(P['resize'])+"),"
			   "CenterCrop("+str(P['resize'])+"),"
			   "ToTensor(),"
                           "Normalize("+str(P['mean_RGB'])+","+str(P['std_RGB'])+")])")
P['mask_transform_test'] = ("Compose(["
			    "Resize("+str(P['resize'])+"),"
			    "CenterCrop("+str(P['resize'])+"),"
			    "ToTensor(),"
			    "])")
#---------------------#
# C) Start training   #
#---------------------#
if __name__ == "__main__":
    # start_with_json(sys.argv[1:], P)
    # or
    trainModel(P)    

   
# start from a json folder:
#--------------------------
#   python main_train.py --jsonfolder ../results/folder_json_torun
