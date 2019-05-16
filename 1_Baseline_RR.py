import scipy.io as sio
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import pandas as pd
import os
from scipy.spatial import distance


'''
META-Definitions
'''
n_pic = 3*180                #Number of pictures
n_ch = 64                   #Number of channels
n_samples = 576             #Number of samples
method = 'feature_vector'   #'ani_vs_inani'  ||  'feature_vector'

'''
Loading data
'''

#Load data 
no_pers = 4; #Number of trials
eeg_all = np.empty((n_pic*no_pers, n_ch*n_samples));
supercat_all = np.empty((n_pic*no_pers, 1));
featurevec_all = np.empty((n_pic*no_pers, 2048));
path = '/Users/Eigil/Documents/EEG/Nicolai/data/'

#OBS: THE PATH SHOULD BE CHANGED ACCORDINGLY

#Loop: Convert all EEG-data into 2D array
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
#    eeg_reshaped = np.empty((n_pic, n_ch*n_samples))
    for i in range(n_pic):
#        print(f,i)
        for j in range(n_ch):
            for k in range(n_samples):
#                eeg_reshaped[i ,k + j*n_samples] = eeg_events[j, k, i]#
                eeg_all[i + (f-1)*n_pic,k+j*n_samples] = eeg_events[j, k, i]


#Creation of image dicts
image_dict = {tuple(image_semantics[i,:]):cat[i] for i in range(len(cat))}
imageid_dict = {tuple(image_semantics[i,:]):imageid[i] for i in range(len(cat))}



#Lowering temporal resolution to 128 Hz (this has been found to yield approximately same results)
resolution = 4
X = eeg_all
X_f = np.copy(X[:, list(range(0, X.shape[1], resolution))])
if method == 'ani_vs_inani':
    y = ani_vs_inani
elif method == 'feature_vector':
    y = featurevec_all



'''
Baseline prediction: Ridge regression
'''

K1 = 10 #Outer loop 
K2 = 10 #Inner loop


CV1 = model_selection.KFold(K1, shuffle=True, random_state=42)
CV2 = model_selection.KFold(K2, shuffle=True, random_state=42)


alpha_ridge = [1, 1e5, 5*1e5, 1e6, 5*1e6, 1e7, 1e8] #Regularization values
# In preliminary analyses, it was found that the optimal alpha-value was centered around 1e6

#Empty arrays
val_error = np.empty((len(alpha_ridge), K2, K1)) #All validation errors
optimal_alpha = np.empty((K1, 1)) #Empty array for optimal alpha values in each outer split
test_error = np.empty((K1, 1))    #Empty array for test errors
classification = np.empty((K1, 1)) #Empty array for classification arrays

k1 = 0
for train_index_o, test_index_o in CV1.split(X_f, y):
    X_train_o = X_f[train_index_o, :]
    y_train_o = y[train_index_o]
    X_test = X_f[test_index_o, :]
    y_test = y[test_index_o]
    
    k2 = 0
    for train_index_i, test_index_i in CV2.split(X_train_o, y_train_o):
        X_train = X_train_o[train_index_i, :]
        y_train = y_train_o[train_index_i]
        X_val = X_train_o[test_index_i, :]
        y_val = y_train_o[test_index_i]
        for i in range(len(alpha_ridge)):
            clf = Ridge(alpha=alpha_ridge[i])
            clf = clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            val_error[i, k2, k1] = mean_squared_error(y_pred, y_val)
        k2 = k2 + 1
        
    optimal_alpha[k1] = alpha_ridge[np.argmin(np.mean(val_error[:, :, k1], axis=1))]
    
    # Training with optimal alpha
    clf = Ridge(alpha = optimal_alpha[k1]).fit(X_train_o, y_train_o)
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

#Saving all results
path2 = '/Users/Eigil/Dropbox/DTU/Advanced Machine Learning/Mind Reading/'
np.save(path2 + 'val_err', val_error)
np.save(path2 + 'test_err', test_error)

np.savetxt(path2 + 'alpha_ridge.csv', alpha_ridge, delimiter=",")
np.savetxt(path2 + 'test_err.csv', test_error, delimiter=",")
np.savetxt(path2 + 'classification_err.csv', classification, delimiter=",")
