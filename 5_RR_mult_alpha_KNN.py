''' 
This script performs multiple ridge regressions, each with an optimal
alpha value. In addition, the KNN-classifier is utilized
'''


import scipy.io as sio
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import pandas as pd
import os
from scipy.spatial import distance
from collections import Counter


#KNN: T

def KNN(y_pred, K, image_semantics, distmetric='euclidean'):
    #KNN:
    # Input: y_pred: Prediced y-vector
    #        K: Number of neighbours
    #        image_semantics: Actual y-vectors (matrix)
    #        distmetric: Distanace metric
    # Output: List of strings of KNN category
    dist_matr = distance.cdist(y_pred, image_semantics[0:180], distmetric)
    dist_matr_index = np.argsort(dist_matr, axis=1)
    index_image_hat = dist_matr_index[:, 0:int(K)]
    hat_cat = np.array([image_semantics[c,:] for c in index_image_hat])
    
    hatcat_lst = []
    hatcat2 = np.empty((hat_cat.shape[0], hat_cat.shape[1]), dtype=object)
    for n in range(hat_cat.shape[0]):
        for k in range(hat_cat.shape[1]):
            hatcat2[n, k] = image_dict[tuple(hat_cat[n, k, :])]
        KNN = hatcat2[n]
        most_common = Counter(KNN).most_common()
        if len(most_common) != 1:
            while most_common[0][1] == most_common[1][1]:
                KNN = KNN[:-1]
                most_common = Counter(KNN).most_common()
                if KNN.shape[0] == 1:
                    break
        pred_label = most_common[0][0]
        hatcat_lst.append(pred_label)
    
    return(np.array(hatcat_lst))


'''
META
'''
n_pic = 3*180                 #Number of pictures
n_ch = 64                   #Number of channels
n_samples = 576             #Number of samples
method = 'feature_vector'   #'ani_vs_inani'  ||  'feature_vector'


channels = [0, 2, 6, 13, 17, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 39, 50, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63] # O1 & O2
    
length_of_channel = 576 - 240

'''
Load data
'''

#Load data 
no_pers = 4;
eeg_all = np.empty((n_pic*no_pers, len(channels)*length_of_channel));
#eeg_all = np.empty((n_pic*no_pers, n_ch*n_samples));
supercat_all = np.empty((n_pic*no_pers, 1));
featurevec_all = np.empty((n_pic*no_pers, 2048));
path = '/Users/Eigil/Documents/EEG/Nicolai/data/'


for f in range(1,no_pers + 1):
    os.chdir(path + "exp" + str(f))
    print(path + "exp" + str(f))
    
    mat = sio.loadmat("eeg_events.mat")
    eeg_events = mat["eeg_events"]
    image_semantics = sio.loadmat("image_semantics.mat")
    image_semantics = image_semantics["image_semantics"].T
    
    file = pd.read_csv(r"image_order.txt", sep='	')
    cat = np.asarray(file.category)
    imageid = np.asarray(file.image_id)
    
    supercat = np.asarray(file.supercategory)
    supercat_set = list(set(supercat))
    supercat_id = np.array([supercat_set.index(i) for i in supercat])
    
    featurevec_all[(f-1)*n_pic:f*n_pic,:] = image_semantics
    supercat_all[(f-1)*n_pic:f*n_pic] = supercat_id.reshape(n_pic,1)
    for i in range(n_pic):
        channel_n = -1
        for j in range(n_ch):
            if j not in channels:
                continue
            else:
                channel_n = channel_n + 1
            for k in range(length_of_channel):
                eeg_all[i + (f-1)*540,k+channel_n*length_of_channel] = eeg_events[j, k, i]



image_dict = {tuple(image_semantics[i,:]):cat[i] for i in range(len(cat))}
imageid_dict = {tuple(image_semantics[i,:]):imageid[i] for i in range(len(cat))}


resolution = 4
X = eeg_all
X_f = np.copy(X[:, list(range(0, X.shape[1], resolution))])
y = featurevec_all


####################################
K1 = 10
K2 = 10
#CV1 = model_selection.KFold(K1, shuffle=True, random_state=42)
CV1 = model_selection.KFold(K1, shuffle=True, random_state=42)
CV2 = model_selection.KFold(K2, shuffle=True, random_state=42)


alpha_ridge = [1e2, 5*1e2, 1e3, 5*1e3, 1e4, 5*1e4, 1e5, 5*1e5, 1e6, 5*1e6, 1e7, 5*1e7, 1e8, 1e9]
K_values = list(range(2, 10))
optimal_configuration = np.empty((K1, 2048))
test_error = np.empty((K1, 1))
classification = np.empty((K1, 1))
classification_val = np.zeros((K2, K1, len(K_values)))

val_error = np.empty((len(alpha_ridge), 2048, K2))

K_opt = np.empty((K1, 1))


k1 = 0
for train_index_o, test_index_o in CV1.split(X_f, y):
    print('k1: '+str(k1))
    X_train_o = X_f[train_index_o, :]
    y_train_o = y[train_index_o]
    X_test = X_f[test_index_o, :]
    y_test = y[test_index_o]
    
    k2 = 0
    for train_index_i, test_index_i in CV2.split(X_train_o, y_train_o):
        print('k2: '+str(k2))
        X_train = X_train_o[train_index_i, :]
        y_train = y_train_o[train_index_i]      
        X_val = X_train_o[test_index_i, :]
        y_val = y_train_o[test_index_i]
        
        #Finding the best alpha configuraiton
        for i in range(len(alpha_ridge)):
            clf = Ridge(alpha=alpha_ridge[i])
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            val_error[i, :, k2] = np.mean((y_pred-y_val)**2, axis=0)
        
        #Using inner optimal configuration, used for KNN
        inner_optimal_configuration_index = np.argmin(val_error[:, :, k2], axis=0)
        inner_optimal_configuration_lst = [alpha_ridge[x] for x in inner_optimal_configuration_index]
        inner_optimal_configuration = np.array(inner_optimal_configuration_lst)
        clf = Ridge(alpha = inner_optimal_configuration).fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        # KNN
        index_image_test=distance.cdist(y_val, image_semantics[0:180],'euclidean').argmin(axis=1)
        test_cat = np.array([image_semantics[c,:] for c in index_image_test])
        test_cat  = [image_dict[tuple(z)] for z in test_cat]
        for i, K in enumerate(K_values):
            output = KNN(y_pred, K, image_semantics)
            result = ( np.array(test_cat)==output ).tolist()
            classification_val[k2, k1, i] = np.mean(result)
        k2 = k2 + 1
    
    optimal_configuration_index = np.argmin(np.mean(val_error, axis=2), axis=0)
    optimal_configuration_lst = [alpha_ridge[x] for x in optimal_configuration_index]
    optimal_configuration[k1, :] = np.array(optimal_configuration_lst)
    classification_means = np.mean(classification_val[:, k1, :], axis=0)
    K_opt[k1] = K_values[np.argmax(classification_means)]
    print('Optimal K-value: '+str(K_opt[k1]))
    print('Mean classification: '+str(max(classification_means)))
    
    # Training with optimal alpha
    clf = Ridge(alpha = optimal_configuration[k1, :]).fit(X_train_o, y_train_o)
    y_pred = clf.predict(X_test)
    test_error[k1] = mean_squared_error(y_test, y_pred)
    
    #Finding classification
    index_image_test=distance.cdist(y_test, image_semantics[0:180],'euclidean').argmin(axis=1)
    test_cat = np.array([image_semantics[c,:] for c in index_image_test])
    test_cat  = [image_dict[tuple(z)] for z in test_cat]
    output = KNN(y_pred, K_opt[k1], image_semantics)
    result = ( np.array(test_cat)==output ).tolist()
    classification[k1] = np.mean(result)
    k1 = k1 + 1
    
##Saving values
path2 = '/Users/Eigil/Dropbox/DTU/Advanced Machine Learning/Mind Reading/'
np.save(path2 + 'val_err', val_error)
np.save(path2 + 'test_err', test_error)
np.save(path2 + 'classification_val', classification_val)

np.savetxt(path2 + 'K_opt.csv', K_opt, delimiter=",")
np.savetxt(path2 + 'alpha_ridge.csv', alpha_ridge, delimiter=",")
np.savetxt(path2 + 'optimal_configuration.csv', optimal_configuration, delimiter=",")
np.savetxt(path2 + 'test_err.csv', test_error, delimiter=",")
np.savetxt(path2 + 'classification_err.csv', classification, delimiter=",")
