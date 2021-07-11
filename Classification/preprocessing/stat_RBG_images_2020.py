# Compute the average pixel and the standard deviation
#------------------------------------------------------

import numpy as np
from PIL import Image
import os
import pandas as pd

class TrainingLoaderSimple():

    def __init__(self, csv_file, path_data):
        self.label_csv = pd.read_csv(csv_file)
        self.path_data = path_data

    def __getitem__(self, idx):
        # the image
        img_name = self.label_csv.iloc[idx, 0] + ".jpg"
        path_img = os.path.join(self.path_data, img_name)
        image = np.array( Image.open(path_img) )
        # the label
        label = self.label_csv.iloc[idx, -1]

        return [image, label, img_name]

    def __len__(self):
        return len(self.label_csv)



class TrainingLoaderSimple_combined_2019_2020():

    def __init__(self, csv_file):
        self.label_csv = pd.read_csv(csv_file)

    def __getitem__(self, idx):
        # the image
        if (self.label_csv['source'][idx] == 'ISIC20'):
            img_name = os.path.join('../../data/ISIC_2020_jpeg/train/', self.label_csv['image_id'][idx] + ".jpg")
        else:
            img_name = os.path.join('../../../melanoma_project_2019/data/data_ISIC_2019/ISIC_2019_Training_Input/', self.label_csv['image_id'][idx] + ".jpg")
        image = np.array( Image.open(img_name) )
        # the label
        label = self.label_csv['target'][idx]

        return [image, label, img_name]

    def __len__(self):
        return len(self.label_csv)

    

# data loader
#------------
csv_file =  '../../data/data_ISIC_combined_2019_2020/folds_13062020.csv'
path_data = '..'
#csv_file =  '../../data/ISIC_2020_jpeg/train.csv'
#path_data = '../../data/ISIC_2020_jpeg/train/'
#csv_file =  '../../data/ISIC_2020_jpeg/test.csv'
#path_data = '../../data/data_ISIC_2020/test/'

#mole_loader_simple = TrainingLoaderSimple(csv_file,path_data)
mole_loader_simple = TrainingLoaderSimple_combined_2019_2020(csv_file)
N = mole_loader_simple.__len__()
df = pd.DataFrame(columns=('nameImage','label','num_loader',
                           'size_x','size_y',
                           'mean_pixel','std_pixel',
                           'mean_pixel_R','std_pixel_R',
                           'mean_pixel_G','std_pixel_G',
                           'mean_pixel_B','std_pixel_B'))
# loop all images
#----------------
for i in range(N):
    print( i )
    # take one image
    img,label,filename = mole_loader_simple.__getitem__(i) # load
    # stat
    nx,ny = img.shape[1],img.shape[0]
    df.loc[i] = [filename, label, i, 
                 nx, ny,
                 img.mean().item(), img.std().item(),
                 img[:,:,0].mean().item(), img[:,:,0].std().item(),
                 img[:,:,1].mean().item(), img[:,:,1].std().item(),
                 img[:,:,2].mean().item(), img[:,:,2].std().item()]

# save tableau
#-------------
#df.to_csv('stat_pixel_train_2020.csv',index=False)
df.to_csv('stat_pixel_train_combined_2019_2020.csv',index=False)

# mean RGB, std RGB (2019)
#-------------------------
# df = pd.read_csv('stat_pixel_all_2020.csv') 
# np.mean(df['mean_pixel_R'])/255
# np.std(df['mean_pixel_R']/255)
# 
#  mean: Red, Green, Blue
# 0.8060787287898581
# 0.5911773606695254
# 0.6210193178753612
#  std:  Red, Green, Blue
# 0.11992922884240897
# 0.16658055488936796
# 0.14301547614528323
# 
# transforms.Normalize([0.80608,0.59118,0.62102],[0.11993,0.16658,0.14302])
#
# 
# mean RGB, std RGB (2019-2020)
#-------------------------
# df = pd.read_csv('stat_pixel_train_combined_2019_2020.csv') 
# np.mean(df['mean_pixel_R'])/255
# np.std(df['mean_pixel_R']/255)
# 
#  mean: Red, Green, Blue
# 0.7469374152241233
# 0.5814371152058563
# 0.5622773126873651
#  std:  Red, Green, Blue
# 0.1502238275425175
# 0.1399492484999321
u# 0.1532718590231839
# 
# transforms.Normalize([0.74694,0.58144,0.56228],[0.15022,0.13995,0.15327])
#








