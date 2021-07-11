# libraries pytorch
import torch
#from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
# classic
import numpy as np
from PIL import Image
import os, random
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from utils.Unet import UNet_1024
import torch.nn.functional as F
import cv2 as cv
import random

## ASSEMBLE THE MODEL AND LOAD WEIGHTS
myUnet = UNet_1024()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myUnet = myUnet.to(device)
myUnet.load_state_dict(torch.load("../../Unet_Weight/myNetwork_UNet1024.pth"))
for param in myUnet.parameters():  # freeze all layers
    param.requires_grad = False

class TrainingLoader():

    def __init__(self, csv_file, path_data, nb_classes,
                 transform_train, transform_test,
                 pct_train_set=.8, shuffle_dataset=True):
        # init
        self.label_csv = pd.read_csv(csv_file)
        self.path_data = path_data
        self.nb_classes = nb_classes
        self.transform_train = transform_train
        self.transform_test = transform_test
        # split train/test set
        N = len(self.label_csv)
        #N = int( len(self.label_csv)/10 ) # when debugging
        indices = list(range(N))
        split = int(np.floor(pct_train_set*N))
        if shuffle_dataset :
            random.shuffle(indices)
        train_indices, test_indices = indices[:split],indices[split:]
        # Creating PT data samplers and loaders:
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

    def __getitem__(self, idx):
        # the image
        if (len(self.label_csv) == 57224): # combined 2019 - 2020
            if (self.label_csv['source'][idx] == 'ISIC20'):
                img_name = os.path.join('../../data/ISIC_2020_jpeg/train/', self.label_csv['image_id'][idx] + ".jpg")
            else:
                img_name = os.path.join('../../data/ISIC_2019_Training_Input/', self.label_csv['image_id'][idx] + ".jpg")
        else:
            img_name = os.path.join(self.path_data, self.label_csv.iloc[idx, 0] + ".jpg")

        image = Image.open(img_name)

        if idx in self.test_sampler:
            image_tensor = self.transform_test(image)
        else:
            image_tensor = self.transform_train(image)

        for_segmentation_image = image_tensor
        for_segmentation_image = F.interpolate(torch.tensor(np.float32(for_segmentation_image.unsqueeze(0))),(400,400),mode='bilinear')
        for_segmentation_image = for_segmentation_image.to(device)
        score = myUnet(for_segmentation_image)
        prediction_vector = (np.argmax(score.data.cpu(), axis=1)).view(-1)
        predicted_mask = prediction_vector.view([1, 1, 400, 400])[0]
        mask = F.interpolate(torch.tensor(np.float32(predicted_mask.unsqueeze(0))),(224,224),mode='bilinear')
        # CONCAT THE IMAGE AND MASK
        concat_image_mask_tensor = torch.cat([image_tensor,mask[0]])
        ## CONCAT PROBABILITY CHANNEL
        
        # the features
        meta_features = create_meta(self.label_csv['age_approx'][idx],
                                    self.label_csv['sex'][idx],
                                    self.label_csv['anatom_site_general_challenge'][idx])
        # the label
        label = self.label_csv['target'][idx]

        return [concat_image_mask_tensor, meta_features, label]

    def __len__(self):
        return len(self.label_csv)

    def compute_pct_classes(self):
        # count the total number of images for each class
        self.sum_train = self.label_csv.iloc[self.train_sampler.indices, range(1,self.nb_classes+1)].sum().values
        N_train = self.sum_train.sum()
        self.pct_train = self.sum_train/N_train
        self.sum_test = self.label_csv.iloc[self.test_sampler.indices, range(1,self.nb_classes+1)].sum().values
        N_test = self.sum_test.sum()
        self.pct_test = self.sum_test/N_test


    
def create_meta(age,sex,anatom_site):
    # sex (2)
    # ['female', 'male']
    # anatom_site (8)   label_csv["anatom_site_general_challenge"].unique() 
    #----------------
    # ['head/neck', 'lateral torso', 'lower extremity', 'oral/genital','palms/soles', 'torso', 'unknown', 'upper extremity']
    # initialize
    meta_features = torch.zeros(11)
    tp = ['head/neck', 'lateral torso', 'lower extremity', 'oral/genital','palms/soles', 'torso', 'unknown', 'upper extremity']
    # age
    meta_features[0] = age
    # sex
    if (sex == 'female'):
        meta_features[1] = 1
    else:
        meta_features[2] = 1
    # position
    idx = tp.index(anatom_site)
    meta_features[3+idx] = 1

    return meta_features