'''
This script solves the ridge regression problem but with a multivariate 
alpha vector
'''


import scipy.io as sio
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import pandas as pd
import os
from scipy.spatial import distance


'''
META
'''
n_pic = 3*180                 #Number of pictures
n_ch = 64                   #Number of channels
n_samples = 576             #Number of samples
method = 'feature_vector'   #'ani_vs_inani'  ||  'feature_vector'


channels = [0, 2, 6, 13, 17, 18, 20, 21, 23, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 39, 50, 53, 55, 56, 57, 58, 59, 60, 61, 62, 63] # O1 & O2


length_of_channel = 576 - 240

'''
Load data
'''

no_pers = 4;
eeg_all = np.empty((n_pic*no_pers, len(channels)*length_of_channel));
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



#Lowering temporal resolution
resolution = 4
X = eeg_all
X_f = np.copy(X[:, list(range(0, X.shape[1], resolution))])
if method == 'ani_vs_inani':
    y = ani_vs_inani
elif method == 'feature_vector':
    y = featurevec_all



'''
Ridge regression with tunable alpha values:
'''
K1 = 10
K2 = 10
CV1 = model_selection.KFold(K1, shuffle=True, random_state=42)
CV2 = model_selection.KFold(K2, shuffle=True, random_state=42)

X_train_o, X_test, y_train_o, y_test = model_selection.train_test_split(X_f, y, test_size=0.2, random_state=1)
alpha_ridge = [1e2, 5*1e2, 1e3, 5*1e3, 1e4, 5*1e4, 1e5, 5*1e5, 1e6, 5*1e6, 1e7, 5*1e7, 1e8, 1e9]
optimal_configuration = np.empty((K1, 2048))
test_error = np.empty((K1, 1))
classification = np.empty((K1, 1))

#val_error = np.empty((len(alpha_ridge), K2, K1))
val_error = np.empty((len(alpha_ridge), 2048, K2))

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
        for i in range(len(alpha_ridge)):
            clf = Ridge(alpha=alpha_ridge[i])
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            val_error[i, :, k2] = np.mean((y_pred-y_val)**2, axis=0)
        k2 = k2 + 1
    
    optimal_configuration_index = np.argmin(np.mean(val_error, axis=2), axis=0)
    optimal_configuration_lst = [alpha_ridge[x] for x in optimal_configuration_index]
    optimal_configuration[k1, :] = np.array(optimal_configuration_lst)
    # Training with optimal alpha
    clf = Ridge(alpha = optimal_configuration[k1, :]).fit(X_train_o, y_train_o)
    y_pred = clf.predict(X_test)
    test_error[k1] = mean_squared_error(y_test, y_pred)
    
    #Finding classification
    index_image_test=distance.cdist(y_test, image_semantics,'euclidean').argmin(axis=1)
    index_image_hat=distance.cdist(y_pred, image_semantics,'euclidean').argmin(axis=1)
    test_cat = np.array([image_semantics[c,:] for c in index_image_test])
    test_cat  = [image_dict[tuple(z)] for z in test_cat]
    hat_cat = np.array([image_semantics[c,:] for c in index_image_hat])
    hat_cat  = [image_dict[tuple(z)] for z in hat_cat ]
    result = ( np.array(test_cat)==np.array(hat_cat) ).tolist()
    classification[k1] = sum(result)/y_pred.shape[0]
    
    k1 = k1 + 1

#Save the optimal val_test_alpha
path2 = '/Users/Eigil/Dropbox/DTU/Advanced Machine Learning/Mind Reading/'
np.save(path2 + 'val_err', val_error)
np.save(path2 + 'test_err', test_error)

np.savetxt(path2 + 'alpha_ridge.csv', alpha_ridge, delimiter=",")
np.savetxt(path2 + 'optimal_configuration.csv', optimal_configuration, delimiter=",")
np.savetxt(path2 + 'test_err.csv', test_error, delimiter=",")
np.savetxt(path2 + 'classification_err.csv', classification, delimiter=",")
