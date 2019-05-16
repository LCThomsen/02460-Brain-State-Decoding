'''
3-Layer ANN with Bayesian optimization. Saves the results as a h5 file for the 
captioning generating network
'''
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, MaxPooling2D, Flatten, Activation
import scipy.io as sio
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
import pandas as pd
import os
from scipy.spatial import distance
from scipy.stats import mode
from sklearn.decomposition import PCA
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold

# =============================================================================
# LOAD DATA
# =============================================================================
# Compute cluster GPU:
os.environ["CUDA_VISIBLE_DEVICES"]="4"

# PATH OF FILES:
#path = '/scratch/Nicolai2/'
path = '/Users/lukasthomsen/Filer:Dokumenter/Advanced machine learning/Project/Nicolai2/' 

# Creation of dictionaries from image semantics to image_id and image_category
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
n_ch = 64                   #Number of channels
n_samples = 576             #Number of samples
method = 'feature_vector'   #'ani_vs_inani'  ||  'feature_vector'

# Choosing which channels to use (from feature selection):
channels = [0,2,6,13,17,18,20,21,22,23,24,25,26,27,28,29,30,31,32,39,50,53,55,56,57,58,59,60,61,62,63]

# Length of each channel (from ffeature selection):
length_of_channel = 576 - 240

# Number of sessions to include
no_pers = 4;

# Reading the data
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
pca = PCA(n_components=300, whiten=True)
X = pca.fit_transform(X)

# =============================================================================
# TRAIN MODEL
# =============================================================================

RANDOM1 = 10;
RANDOM = 141;

(X1, X2, y1, y2)=model_selection.train_test_split(X, y, test_size=0.1,random_state=RANDOM1)

def get_model(dense1=512,dense2=512,dropout2_rate=0.5,dropout1_rate=0.5,input_shape=1):
    model = Sequential()
    model.add(Dense(int(dense1), input_dim =input_shape, activation = 'relu'))
    model.add(Dropout(dropout1_rate, name="dropout_1"))
    model.add(Dense(int(dense2), activation = 'relu'))
    model.add(Dropout(dropout2_rate, name="dropout_2"))
    model.add(Dense(2048,activation = 'linear'))
    return model

def fit_with(dense1,dense2,dropout1_rate,dropout2_rate):

    # Create the model using a specified hyperparameters
    splits = 3
    cv = KFold(n_splits=splits, random_state=RANDOM, shuffle=True)
    score = np.zeros(splits)
    count = 0
    for train_index, test_index in cv.split(X1):
        x_train, x_test, y_train, y_test = X1[train_index], X1[test_index], y1[train_index], y1[test_index]

        input_shape = x_train.shape[1]
        model = get_model(dense1,dense2,dropout1_rate,dropout2_rate,input_shape)
        
        keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
        model.compile(loss='mse', metrics=['mse'],
                      optimizer='RMSprop')

        model.fit(x_train,y_train, epochs=100, batch_size=32, verbose=0) 
        r = model.evaluate(x_test,y_test,steps=30, verbose=0)
        score[count] = r[0]
        count = count + 1

    return np.mean(-score)


from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'dense1': (10,2048),
           'dense2': (10,2048),
           'dropout1_rate': (0.01, 0.99),
           'dropout2_rate': (0.01, 0.99)}

optimizer = BayesianOptimization(
    f=fit_with,
    pbounds=pbounds,
    verbose=2,
    )

optimizer.maximize(init_points=100, n_iter=400,)
print('Best parameters:')
print(optimizer.max)
params = optimizer.max['params']

# =============================================================================
# Testing the model on test set:
# =============================================================================
(X1, X2, y1, y2)=model_selection.train_test_split(X, y, test_size=0.1,random_state=RANDOM1)
params['input_shape'] = X1.shape[1]
MODEL = get_model(**params)
MODEL.compile(loss='mse', metrics=['mse'],
          optimizer='RMSprop')
MODEL.fit(X1,y1,epochs=100,batch_size=32)

print('-'*53)
print('Final Results')
y_pred= MODEL.predict(X2)

# =============================================================================
# Classification & MSE
# =============================================================================

#k-nearest neighbors
def KNN(y_pred, K, image_semantics, distmetric='euclidean'):
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

from scipy.spatial import distance
index_image_test=distance.cdist(y2, image_semantics,'euclidean').argmin(axis=1)
index_image_hat=distance.cdist(y_pred, image_semantics,'euclidean').argmin(axis=1)

test_cat = np.array([image_semantics[c,:] for c in index_image_test])
test_id = [imageid_dict[tuple(z)] for z in test_cat]
test_cat  = [image_dict[tuple(z)] for z in test_cat]


hat_cat = np.array([image_semantics[c,:] for c in index_image_hat])
hat_id = [imageid_dict[tuple(z)] for z in hat_cat ]
hat_cat  = [image_dict[tuple(z)] for z in hat_cat ]

from collections import Counter

words = hat_cat
print(Counter(words).keys()) # equals to list(set(words))
print(Counter(words).values()) # counts the elements' frequency

words = test_cat
print(Counter(words).keys()) # equals to list(set(words))
print(Counter(words).values()) # counts the elements' frequency

results = ( np.array(test_cat)==np.array(hat_cat) ).tolist()
classification=sum(results)/y_pred.shape[0]
print('Classification accuracy')
print(classification)
print('Mean squared error:')
print(mean_squared_error(y_pred,y2))

index_image_test=distance.cdist(y2, image_semantics[0:180],'euclidean').argmin(axis=1)
test_cat = np.array([image_semantics[c,:] for c in index_image_test])
test_cat  = [image_dict[tuple(z)] for z in test_cat]
output = KNN(y_pred, 4, image_semantics)
result = ( np.array(test_cat)==output ).tolist()
classification = np.mean(result)
print('Classification accuracy 4-NN')
print(classification)


# =============================================================================
# SAVE as H5
# =============================================================================

y_hat = y_pred.tolist()
y_test = y2.tolist()
hat_cat = np.array([image_semantics[c,:] for c in index_image_hat])
test_cat = np.array([image_semantics[c,:] for c in index_image_test])
#
import h5py
f = h5py.File("/Users/lukasthomsen/Filer:Dokumenter/Advanced machine learning/Project/data/val.h5",'w')
h = h5py.File("/Users/lukasthomsen/Filer:Dokumenter/Advanced machine learning/Project/data/test.h5",'w')
count = 0
for i in range(len(y_hat)):
    count = count +1
    try:
        f[str(imageid_dict[tuple(test_cat[i,:])])] = hat_cat[i,:] 
        h[str(imageid_dict[tuple(test_cat[i,:])])] = test_cat[i,:]
    except:
        continue
f.close()
h.close()

