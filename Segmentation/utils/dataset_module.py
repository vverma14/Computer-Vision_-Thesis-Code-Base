# libraries pytorch
import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
# classic
import numpy as np
from PIL import Image
import random, glob, os
import numpy as np
#from skimage import measure     # identify connected components

##################################
# Segmentation Loader
##################################	
class Mole_dataset():

    def __init__(self, path_images, path_masks,
                 img_transform_train, mask_transform_train,
                 img_transform_test, mask_transform_test,
                 pct_train_set=None, shuffle_dataset=True):
        # setup path+names
        self.path_images = path_images
        self.path_masks = path_masks
        filenames_tp = [os.path.basename(x) for x in sorted(glob.glob(path_images+"*.jpg"))]
        self.filenames = [f[:-4] for f in filenames_tp] # remove ".jpg"
        # setup the transformation
        self.img_transform_train = img_transform_train
        self.mask_transform_train = mask_transform_train
        self.img_transform_test = img_transform_test
        self.mask_transform_test = mask_transform_test
        # split train/test set
        if (pct_train_set != None):
            dataset_size = len(self.filenames)
            indices = list(range(dataset_size))
            split = int(np.floor(pct_train_set * dataset_size))
            if shuffle_dataset :
                random.shuffle(indices)
            train_indices, test_indices = indices[:split], indices[split:]
            # Creating PT data samplers and loaders:
            self.train_sampler = SubsetRandomSampler(train_indices)
            self.test_sampler = SubsetRandomSampler(test_indices)

    def __getitem__(self, idx):
        # get image
        filename = self.filenames[idx]
        with open(self.path_images+filename+'.jpg', 'rb') as f:
            image = Image.open(f).convert('RGB')
        with open(self.path_masks+filename+ '_Segmentation.png', 'rb') as f:
            mask = Image.open(f).convert('P')
        # apply transformation
        seed = np.random.randint(0,40000) # use the same seed to img and mask
        if idx in self.train_sampler:
            random.seed(seed)
            image_tensor = self.img_transform_train(image)
            random.seed(seed)
            mask_tensor = self.mask_transform_train(mask)
        else:
            random.seed(seed)
            image_tensor = self.img_transform_test(image)
            random.seed(seed)
            mask_tensor = self.mask_transform_test(mask)
        # make the mask an long 'int'
        mask_tensor_long = (mask_tensor>1e-10).long()
        
        return image_tensor, mask_tensor_long

    def __len__(self):
        return len(self.filenames)

    
class Mole_dataset_simple():

    def __init__(self, path_images, path_masks):
        self.path_images = path_images
        self.path_masks = path_masks
        filenames_tp = [os.path.basename(x) for x in sorted(glob.glob(path_images+"*.jpg"))]
        self.filenames = [f[:-4] for f in filenames_tp] # remove ".jpg"
        self.toTensor = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        # get image
        filename = self.filenames[idx]
        with open(self.path_images+filename+'.jpg', 'rb') as f:
            image = Image.open(f).convert('RGB')
            image_tensor = self.toTensor(image)
        with open(self.path_masks+filename+ '_Segmentation.png', 'rb') as f:
            mask = Image.open(f).convert('P')
            mask_tensor = self.toTensor(mask)

        return [image_tensor, mask_tensor, filename]

    def __len__(self):
        return len(self.filenames)
