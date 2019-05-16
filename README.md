README

### 02460-Brain-State-Decoding ###
### Code submitted for the course 02460 Advanced Machine Learning ###


Most of the code acts as standalone files. Much of the analysis have been conducted by taking the source code and by changing some parameter (e.g. inclusion of the KNN method, which channels and times to consider). Not all permutations have been included, as this cod represent the main ideas of the analysis.

The code is structured as follows:
Most codes contains loading, 10x10 nested Cross Validation and saving

!!! MAIN CODE!!! :

[1] Baseline_RR.py: 
Baseline Ridge Regression with all channels and time signals. 


[2] Backwards-Feature-Selection.py: 
Backwards Channel Feature Selection, based on baseline method


[3] Feature_selection_channel_length.py:
Exhaustive channel length search, based on baseline method.


[4] RR_mult_alpha.py:
Ridge regression with 2048 independent alphas.


[5] RR_mult_alpha_KNN.py:
Ridge regression with 2048 independent alphas and a KNN-classifier. Is included to show how utilized KNN works in this study


[6] NN_BO.py:
ANN with Bayesian optimisation


[7] T-SNE.py:
T-sne plot of predictions and test image semantics


[8] CCA_and_PCA.R
Conducts the CCA and PCA analysis

[9] predict.py:
Code from Nicolai Pedersen. Caption generating network. 



!!! SNIPPETS !!!

[A] DTU_HPC_example:
Standard form of how code was submitted to the DTU HPC Cluster

[B] KNN_LOO:
Contains a Leave-One-Out Cross Validation of the KNN method with relation to the image semantics. Is included to show the perfect clustering

[C] Mean_response
Small snippet used to show how the X- and y-matrices are handled under leaned response

