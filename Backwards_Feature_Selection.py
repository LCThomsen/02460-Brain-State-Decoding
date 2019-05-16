'''
DESCRIPTION:
This code runs a backwards feature selection with respect to channels, starting
with 64 channels and terminating when further neglect of channels lead to lower
validation error
'''

import scipy.io as sio
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import pandas as pd
import os
from scipy.spatial import distance
from scipy.stats import mode

'''
META
'''
n_pic = 3*180                 #Number of pictures
n_ch = 64                   #Number of channels
n_samples = 576             #Number of samples
method = 'feature_vector'   #'ani_vs_inani'  ||  'feature_vector'

'''
Load data
'''
no_pers = 4;
eeg_all = np.empty((n_pic*no_pers, n_ch*n_samples));
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
    
    file = pd.read_csv(r"image_order.txt", sep='\t')
    cat = np.asarray(file.category)
    imageid = np.asarray(file.image_id)
    
    supercat = np.asarray(file.supercategory)
    supercat_set = list(set(supercat))
    supercat_id = np.array([supercat_set.index(i) for i in supercat])
    
    featurevec_all[(f-1)*n_pic:f*n_pic,:] = image_semantics
    supercat_all[(f-1)*n_pic:f*n_pic] = supercat_id.reshape(n_pic,1)
    for i in range(n_pic):
        for j in range(n_ch):
            for k in range(n_samples):
                eeg_all[i + (f-1)*n_pic,k+j*n_samples] = eeg_events[j, k, i]
                
                
image_dict = {tuple(image_semantics[i,:]):cat[i] for i in range(len(cat))}
imageid_dict = {tuple(image_semantics[i,:]):imageid[i] for i in range(len(cat))}

print('Loaded data succesfully')


#Lowering temporal resolution
resolution = 4
X = eeg_all
X_f = np.copy(X[:, list(range(0, X.shape[1], resolution))])
length_channel = n_samples/resolution
if length_channel.is_integer() != True:
    print('WARNING: Resolution and number of samples are not compatible')
    exit()
length_channel = int(length_channel)
y = featurevec_all


'''
Initialization
'''
####################
optimal_alpha_list = []
feature_err = []
classification_err = []
feature_rmv = [float('nan')] #First nan, as no features are removed initially
feature_selection = list(range(0, n_ch)) #Selected features
index_list = list(range(0, n_ch*length_channel)) #Indices of remaining channels
index_rm_list = [] #List of removed indices
####################
K1 = 10
K2 = 10
CV1 = model_selection.KFold(K1, shuffle=True, random_state=42)
CV2 = model_selection.KFold(K2, shuffle=True, random_state=42)

# The regularization is effectively a window, that allows for subsequently
# easing of the regularization, as the feature selection progresses
alpha_10_lower= 5;
alpha_ridge = np.logspace(alpha_10_lower, alpha_10_lower+2, 6) #[1e5, 1e6, 1e7]
####################


print('Beginning initial test')

converged = False
it = 0
max_iter = 64


X_slct = X_f[:, index_list]


#Validation error, optimal alpha, test error and classification
val_error = np.empty((len(alpha_ridge), K2, K1))
optimal_alpha_lst = np.empty((K1, 1))
test_error = np.empty((K1, 1))
classification = np.empty((K1, 1))


# Initial model
k1 = 0
for train_index_o, test_index_o in CV1.split(X_slct, y):
    X_train_o = X_slct[train_index_o, :]
    y_train_o = y[train_index_o]
    X_test = X_slct[test_index_o, :]
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
        
    optimal_alpha_lst[k1] = alpha_ridge[np.argmin(np.mean(val_error[:, :, k1], axis=1))]
    # Training with optimal alpha
    clf = Ridge(alpha = optimal_alpha_lst[k1]).fit(X_train_o, y_train_o)
    y_pred = clf.predict(X_test)
    test_error[k1] = mean_squared_error(y_test, y_pred)
    print('Test error'+str(test_error[k1]))
    
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

previous_model_test_error = np.mean(test_error)
print('Initial model error: ' + str(previous_model_test_error))

feature_err.append(previous_model_test_error)
classification_err.append(np.mean(classification))
optimal_alpha = mode(optimal_alpha_lst)[0][0]
alpha_used_index = np.where((optimal_alpha == alpha_ridge))[0][0]
if alpha_used_index == 0:
    alpha_10_lower = alpha_10_lower - 1
    alpha_ridge = np.logspace(alpha_10_lower, alpha_10_lower+2, 3)
elif alpha_used_index == 2:
    alpha_10_lower = alpha_10_lower + 1
    alpha_ridge = np.logspace(alpha_10_lower, alpha_10_lower+2, 3)
        

optimal_alpha_list.append(optimal_alpha)


#Feature selection loop
print('Beginning feature selection')
while converged != True and it < max_iter:
    it = it + 1
    print('Iteration number: ' + str(it))
    error_lst = np.empty((len(feature_selection), 1))
    for i1 in range(len(feature_selection)):
        X_try = np.concatenate((X_slct[:, 0:i1*length_channel], X_slct[:, (i1+1)*length_channel:]), axis=1)
        val_error = list(range(0, K1))
        i2 = 0
        for train_index, test_index in CV1.split(X_try, y):
            X_train = X_try[train_index, :]
            y_train = y[train_index]
            X_test = X_try[test_index, :]
            y_test = y[test_index]
            clf = Ridge(alpha=optimal_alpha).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            val_error[i2] = mean_squared_error(y_pred, y_test)
            i2 = i2 + 1
        error_lst[i1] = np.mean(val_error)
        
    lowest_err = min(error_lst)
    lowest_err_index = np.argmin(error_lst)
    
    if previous_model_test_error < lowest_err:
        print('Found optimal model in a backwards ff setting:')
        print('Resulting channels:')
        for channel in feature_selection:
            print(channel)
        converged = True
    else:
        index_rm = feature_selection[lowest_err_index]
        print('Removing channel with index '+str(index_rm))
        feature_selection.remove(index_rm)
        feature_rmv.append(index_rm)
        index_rm_list.extend(list(range(index_rm*length_channel, (index_rm+1)*length_channel)))
        index_list = [index for index in index_list if index not in index_rm_list]
        X_slct = X_f[:, index_list]
                        
        X_train_o, X_test, y_train_o, y_test = model_selection.train_test_split(X_slct, y, test_size=0.2, random_state=1)
        
        val_error = np.empty((len(alpha_ridge), K2, K1))
        optimal_alpha_lst = np.empty((K1, 1))
        test_error = np.empty((K1, 1))
        classification = np.empty((K1, 1))
        
        k1 = 0
        for train_index_o, test_index_o in CV1.split(X_slct, y):
            X_train_o = X_slct[train_index_o, :]
            y_train_o = y[train_index_o]
            X_test = X_slct[test_index_o, :]
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
                
            optimal_alpha_lst[k1] = alpha_ridge[np.argmin(np.mean(val_error[:, :, k1], axis=1))]
            # Training with optimal alpha
            clf = Ridge(alpha = optimal_alpha_lst[k1]).fit(X_train_o, y_train_o)
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
        
        previous_model_test_error = np.mean(test_error)
        print('Test error: '+str(previous_model_test_error))
        feature_err.append(previous_model_test_error)
        classification_err.append(np.mean(np.mean(classification)))
        optimal_alpha = mode(optimal_alpha_lst)[0][0]
        optimal_alpha_list.append(optimal_alpha)
        
        alpha_used_index = np.where((optimal_alpha == alpha_ridge))[0][0]
        if alpha_used_index == 0:
            alpha_10_lower = alpha_10_lower - 1
            alpha_ridge = np.logspace(alpha_10_lower, alpha_10_lower+2, 3)
        elif alpha_used_index == 2:
            alpha_10_lower = alpha_10_lower + 1
            alpha_ridge = np.logspace(alpha_10_lower, alpha_10_lower+2, 3)


#Save the optimal val_test_alpha
df = pd.DataFrame(list(zip(feature_rmv, feature_err, classification_err, optimal_alpha_list)),
                  columns=['Removed features', 'Test error', 'Classification error', 'Optimal alpha'])
    
path2 = '/Users/Eigil/Dropbox/DTU/Advanced Machine Learning/Mind Reading/'

df.to_csv(path2+'Feature_Selection_alpha_window.csv', header=True, index=None)
