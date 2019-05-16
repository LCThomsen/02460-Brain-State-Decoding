''' 
This shows how to mean response after having loaded the data
'''

y_unique = y[0:180]
X_f_mean = np.zeros((int(X_f.shape[0]/12), X_f.shape[1]))


for i in range(len(y_unique)):
    for j in range(len(y)):
        if np.all(y[j, :] == y_unique[i, :]):
            X_f_mean[i, :] += X_f[j, :]
X_f_mean = np.divide(X_f_mean, 12)
