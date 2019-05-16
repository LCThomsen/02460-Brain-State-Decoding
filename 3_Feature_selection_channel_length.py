'''
DESCRIPTION:
This code runs a backwards feature selection with respect to channels length, starting
from the back and terminating until all features are exhausted
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

path = '/Users/lukasthomsen/Filer:Dokumenter/Advanced machine learning/Project/Nicolai2/'
os.chdir(path + "exp" + str(1))
mat = sio.loadmat("eeg_events.mat")
eeg_events = mat["eeg_events"]
image_semantics = sio.loadmat("image_semantics.mat")
image_semantics = image_semantics["image_semantics"].T
file = pd.read_csv(r"image_order.txt", sep='	')
cat = np.asarray(file.category)
imageid = np.asarray(file.image_id)
image_dict = {tuple(image_semantics[i,:]):cat[i] for i in range(len(cat))}
imageid_dict = {tuple(image_semantics[i,:]):imageid[i] for i in range(len(cat))}

n_pic = 3*180                 #Number of pictures
                   #Number of channels
n_samples = 576             #Number of samples
method = 'feature_vector'   #'ani_vs_inani'  ||  'feature_vector'

channels = [0,2,6,13,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,39,50,53,55,56,57,58,59,60,61,62,63]
length_of_channel = 576
n_ch = len(channels)

no_pers = 4;
eeg_all = np.empty((n_pic*no_pers, len(channels)*length_of_channel));
supercat_all = np.empty((n_pic*no_pers, 1));
featurevec_all = np.empty((n_pic*no_pers, 2048));
#path = '/Nicolai2/'
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

X = eeg_all
y = featurevec_all

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

####################

# Length of segment that is cut for every iteration
cut_real = 6*4
cut = int(cut_real / resolution)

optimal_alpha_list = []
feature_err = []
classification_err = []
feature_rmv = np.empty(0,int) #First nan, as no features are removed initially

channel_bits = X_f.shape[1]/n_ch/cut

index_list = list(range(0, n_ch*length_channel))
index_rm_list = []
####################
K1 = 10
K2 = 10
CV1 = model_selection.KFold(K1, shuffle=True, random_state=42)
CV2 = model_selection.KFold(K2, shuffle=True, random_state=42)
#alpha_ridge = [1, 1e4, 1e5, 1e6, 1e7]
alpha_10_lower= 5;
alpha_ridge = np.logspace(alpha_10_lower, alpha_10_lower+2, 3) #[1e5, 1e6, 1e7]
####################


print('Beginning initial test')
converged = False
it = 0
max_iter = 2


X_slct = X_f[:, index_list]


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



print('Beginning feature selection')
error_lst = np.empty((int(channel_bits), 1))    
for i1 in range(1,int(channel_bits)):
#        X_try = np.concatenate((X_slct[:, 0:i1*length_channel], X_slct[:, (i1+1)*length_channel:]), axis=1)¨
    index_list_select = np.empty(0,int)
    new_channel_length=(n_samples/resolution)
    for i in index_list[1:]:
        if i % new_channel_length == 0:
            index_list_select = np.append(index_list_select,index_list[int(i-(new_channel_length)):i-(cut*i1)])
    feature_rmv=np.append(feature_rmv,cut*i1*resolution)

    X_slct = X_f[:, index_list_select.astype(int)]                        
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

#Save the performace as a function of different channel lengths
df = pd.DataFrame(list(zip(feature_rmv, feature_err, classification_err, optimal_alpha_list)),
                  columns=['Removed feature length', 'Test error', 'Classification error', 'Optimal alpha'])
path2 = '/Users/lukasthomsen/Filer:Dokumenter/Advanced machine learning/Project/Nicolai2/'
df.to_csv(path2+'Feature_Selection_alpha_window.csv', header=True, index=None)
