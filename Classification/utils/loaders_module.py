# libraries pytorch
import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
# classic
import numpy as np
from PIL import Image
import os, random
import pandas as pd

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
        # the features
        meta_features = create_meta(self.label_csv['age_approx'][idx],
                                    self.label_csv['sex'][idx],
                                    self.label_csv['anatom_site_general_challenge'][idx])
        # the label
        label = self.label_csv['target'][idx]

        return [image_tensor, meta_features, label]

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
