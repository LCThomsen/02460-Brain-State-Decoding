
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.spatial import distance
from collections import Counter
import seaborn as sns

'''
META
'''
n_pic = 3*180                 #Number of pictures
n_ch = 64                   #Number of channels
n_samples = 576             #Number of samples

    

path = '/Users/Eigil/Documents/EEG/Nicolai/data/exp1/'
os.chdir(path)
image_semantics = sio.loadmat("image_semantics.mat")
image_semantics = image_semantics["image_semantics"].T

file = pd.read_csv(r"image_order.txt", sep='	')
cat = np.asarray(file.category)

y= image_semantics
cat_unique = cat[0:180]


K_values = list(range(2, 30))


y_unique = y[0:180]



dist_matr = distance.cdist(y_unique, y_unique, 'cosine')
sns.heatmap(dist_matr)
plt.show()

dist_matr_index = np.argsort(dist_matr, axis=1)
sns.heatmap(dist_matr_index)
plt.show()


classification_train = np.zeros((180, len(K_values)), dtype=bool)
for i, K in enumerate(K_values):
    print('Running K-value: '+str(K))
    for n in range(180):
        KNN = cat_unique[dist_matr_index[n, 1:(K+1)]]
        most_common = Counter(KNN).most_common()
        #
        if len(most_common) != 1:
            while most_common[0][1] == most_common[1][1]:
                KNN = KNN[:-1]
                most_common = Counter(KNN).most_common()
                if KNN.shape[0] == 1:
                    break
        
        pred_label = most_common[0][0]
        real_label = cat_unique[n]
        classification_train[n, i] = (pred_label == real_label) 
        
print('Accuracies for the K-values')
print(np.mean(classification_train, axis=0))
####
