
from sklearn.decomposition import PCA
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as plot

y_test = y2
y_pred = y_pred

# Plotting options
sns.set()
params = {'legend.fontsize': 10,
          'legend.handlelength': 2}
plot.rcParams.update(params)
import matplotlib.cm as cm
cats = ['airplane', 'elephant', 'pizza', 'sheep', 'train', 'zebra']
colors = cm.rainbow(np.linspace(0, 1, len(cats)))
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)

# =============================================================================
# # 2D TSNE
# =============================================================================
y_tsne = TSNE(n_components=2,perplexity=10).fit_transform(y_pred)
y_tsne_test = TSNE(n_components=2,perplexity=10).fit_transform(y_test)
for j,cat in enumerate(cats):
    index = [i for i,x in enumerate(test_cat) if x == cat]
    y_t = y_tsne[index,:]
    y_t2 = y_tsne_test[index,:]
    ax.scatter(y_t[:,0],y_t[:,1],s = 60,label=cat +' pred' ,alpha=1,marker='1', color = colors[j])
    ax.scatter(y_t2[:,0],y_t2[:,1],s=50,label=cat,color = colors[j])
    ax.grid()
    ax.legend(loc='upper right')
fig.savefig('samplefigures' + str(i),  bbox_inches='tight')


# =============================================================================
# 3D PCA plot
# =============================================================================

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
y_pca = PCA(n_components=3).fit_transform(y_pred)
y_pca_test = PCA(n_components=3).fit_transform(y_test)

#y_tsne = TSNE(n_components=2).fit_transform(y_pred)
#y_tsne = TSNE(n_components=2).fit_transform(y_test)
cats = ['airplane', 'elephant', 'pizza', 'sheep', 'train', 'zebra']
#index = np.empty((6,len(test_cat)))
for j,cat in enumerate(cats):
    index = [i for i,x in enumerate(test_cat) if x == cat]
    y_t = y_pca_test[index,:]
    ax.scatter(y_t[:,0],y_t[:,1],y_t[:,2],label=cat +' pred')
    y_t = y_pca[index,:]
    ax.scatter(y_t[:,0],y_t[:,1],y_t[:,2],label=cat)
    ax.legend()
#
