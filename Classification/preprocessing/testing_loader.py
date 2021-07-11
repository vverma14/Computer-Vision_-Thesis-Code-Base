# Test the loader
#----------------

from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import sys,os
sys.path.append( '../' )
from utils.loaders_module import TrainingLoader

# parameters
transformations = ("transforms.Compose(["
                   "transforms.Pad((0,75)),"
		   #"transforms.CenterCrop(224),"
		   "transforms.RandomHorizontalFlip(),"
                   "transforms.RandomRotation(180),"
                   "transforms.Resize(224),"
		   "transforms.ToTensor()])")
#                   "transforms.Normalize([0.764, 0.546, 0.571], [0.0982, 0.0857, 0.0950])])")
transformations2 = ("transforms.Compose(["
                    "transforms.RandomHorizontalFlip(),"
                    "transforms.RandomRotation(360),"
                    "transforms.RandomResizedCrop(224,scale=(0.8, 1.0)),"
                    "transforms.ColorJitter(brightness=.1, contrast=.1, saturation=.1, hue=.1),"
		    "transforms.ToTensor()])")
transformations0 = ("transforms.Compose(["
		    "transforms.ToTensor()])")

                    #"transforms.Normalize([0.764, 0.546, 0.571], [0.0982, 0.0857, 0.0950])])")
# data loader
#------------
#csv_file = '../../data/mini_data/mini_training_groundTruth_2018/label_mini_2018.csv'
#path_data = '../../data/mini_data/mini_training_input_2018/'
csv_file = '../../data/ISIC2018/ISIC2018_Task3_Training_GroundTruth/ISIC2018_training.csv'
path_data = '../../data/ISIC2018/ISIC2018_Task3_Training_Input/'
mole_loader = TrainingLoader(csv_file,path_data,eval(transformations))
mole_loader2 = TrainingLoader(csv_file,path_data,eval(transformations2))
mole_loader0 = TrainingLoader(csv_file,path_data,eval(transformations0))

# testing
img_1,label_1 = mole_loader.__getitem__(123)
img_2,label_2 = mole_loader0.__getitem__(123)
# img_2-img_1
plt.figure(1)
plt.subplot(1,2,1)
img_1_toPlot = np.swapaxes(img_1.data.numpy(),0,2)
img_2_toPlot = np.swapaxes(img_2.data.numpy(),0,2)
plt.imshow(img_1_toPlot)
plt.subplot(1,2,2)
plt.imshow(img_2_toPlot)
plt.savefig("bidon.pdf")
#plt.show()
