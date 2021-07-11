# compute the proportion of each class

import pandas as pd
import numpy as np

# import
#df_train = pd.read_csv('../../data/data_ISIC_2020/ISIC_2020_jpeg/train.csv')
df_train = pd.read_csv('../../data/data_ISIC_combined_2019_2020/folds_13062020.csv')
#df_test = pd.read_csv('../../data/ISIC_2020_jpeg/test.csv')

# stat train
N = len(df_train)                     # number of images 33,126
melanom = np.sum(df_train['target'])
benign = N-melanom
p = melanom/N
q = benign/N

# training set 2020
#------------------
#                 sum    proportion    weight
# Benign          584       .01763      .98237     
# Melanoma      32542       .98237      .01763
# total         33126                    


# training set 2019-2020
#-----------------------
#                 sum    proportion    weight
# Benign         4922       .08601      .91399     
# Melanoma      52302       .91399      .08601
# total         57224                    


