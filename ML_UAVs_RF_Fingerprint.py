import scipy
from scipy import io
import numpy as np
from google.colab import drive
drive.mount('/content/drive')

drone9 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/v1green_surround_2_RX1.mat')
features = drone9['matrix']
v1 = pd.DataFrame(features)

drone10 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/v1green_surround_2_RX2.mat')
features = drone10['matrix']
v2 = pd.DataFrame(features)

drone11 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/v1green_surround_RX1.mat')
features = drone11['matrix']
v3 = pd.DataFrame(features)

drone12 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/v1green_surround_RX2.mat')
features = drone12['matrix']
v4 = pd.DataFrame(features)

drone21 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/v1green_straight_2_RX1.mat')
features = drone21['matrix']
v5 = pd.DataFrame(features)

drone22 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/v1green_straight_2_RX2.mat')
features = drone22['matrix']
v6 = pd.DataFrame(features)

drone23 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/v1green_straight_RX1.mat')
features = drone23['matrix']
v7 = pd.DataFrame(features)

drone24 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/v1green_straight_RX2.mat')
features = drone24['matrix']
v8 = pd.DataFrame(features)

########################################################################

drone1 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c1yellow_surround_2_RX1.mat')
features = drone1['matrix']
c1 = pd.DataFrame(features)

drone2 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c1yellow_surround_2_RX2.mat')
features = drone2['matrix']
c2 = pd.DataFrame(features)

drone3 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c1yellow_surround_RX2.mat')
features = drone3['matrix']
c3 = pd.DataFrame(features)

drone4 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c2yellow_green_surroud_2_RX1.mat')
features = drone4['matrix']
c4 = pd.DataFrame(features)

drone5 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c2yellow_green_surroud_2_RX2.mat')
features = drone5['matrix']
c5 = pd.DataFrame(features)

drone6 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c2yellow_green_surroud_RX1.mat')
features = drone6['matrix']
c6 = pd.DataFrame(features)

drone7 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c2yellow_green_surroud_RX1.mat')
features = drone7['matrix']
c7 = pd.DataFrame(features)

drone8 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c2yellow_green_surroud_RX2.mat')
features = drone8['matrix']
c8 = pd.DataFrame(features)

##############################################################################################
drone13 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c1yellow_straight_2_RX1.mat')
features = drone13['matrix']
c9 = pd.DataFrame(features)

drone14 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c1yellow_straight_2_RX2.mat')
features = drone14['matrix']
c10 = pd.DataFrame(features)

drone15 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c1yellow_straight_RX1.mat')
features = drone15['matrix']
c11 = pd.DataFrame(features)

drone16 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c1yellow_straight_RX2.mat')
features = drone16['matrix']
c12 = pd.DataFrame(features)

drone17 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c2yellow_green_straight_2_RX1.mat')
features = drone17['matrix']
c13 = pd.DataFrame(features)

drone18 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c2yellow_green_straight_2_RX2.mat')
features = drone18['matrix']
c14 = pd.DataFrame(features)

drone19 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c2yellow_green_straight_RX1.mat')
features = drone19['matrix']
c15 = pd.DataFrame(features)

drone20 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/c2yellow_green_straight_RX2.mat')
features = drone20['matrix']
c16 = pd.DataFrame(features)

############################################################
drone25 = scipy.io.loadmat('drive/My Drive/SongLab/data/New_Data/whitenoise_8332_4096.mat')
features = drone25['n']
none1 = pd.DataFrame(features)

def stack(a,b):
  c = np.vstack((a,b))
  return c

c = stack(c1,c2)
c = stack(c,c3)
c = stack(c,c4)
c = stack(c,c5)
c = stack(c,c6)
c = stack(c,c7)
c = stack(c,c8)
c = stack(c,c9)
c = stack(c,c10)
c = stack(c,c11)
c = stack(c,c12)
c = stack(c,c13)
c = stack(c,c14)
c = stack(c,c15)
c = stack(c,c16)

##############################################

v = stack(v1,v2)
v = stack(v,v3)
v = stack(v,v4)
v = stack(v,v5)
v = stack(v,v6)
v = stack(v,v7)
v = stack(v,v8)

#################################################
#none1

##########################################
#label of none = 0, label of control = 1, label of vedio = 2
y0 = np.zeros((none1.shape[0],1)).astype(int)
y1 = np.ones((c.shape[0],1)).astype(int)
y2 = 2 * np.ones((v.shape[0],1)).astype(int)

##########################
d0 = np.hstack((none1,y0))
d1 = np.hstack((c,y1))
d2 = np.hstack((v,y2))

d = np.vstack((d0,d1))
d = np.vstack((d,d2))

np.random.shuffle(d)

x = d[:,0:-1]
y = d[:,-1].astype(int)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
#x_train = x_train[:,0:-1]

from sklearn.preprocessing import StandardScaler,MinMaxScaler
x_train = MinMaxScaler().fit_transform(x_train)
x_test = MinMaxScaler().fit_transform(x_test)

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
#ss = MinMaxScaler()
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=0.92)
pca.fit(x_train)
x_train = pca.transform(x_train)  
x_test = pca.transform(x_test)

from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train, y_train)
test = knn.predict(x_test)
knn.score(x_test, y_test)

from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='adam', alpha=0.4, hidden_layer_sizes=(30,21,18,15,9,6),max_iter=500)
clf.fit(x_train, y_train)
test = clf.predict(x_test)
clf.score(x_test, y_test)

from sklearn.svm import SVC
svm = SVC(gamma = 'auto')
svm.fit(x_train, y_train)
test = svm.predict(x_test)
svm.score(x_test, y_test)

import sklearn
from sklearn.tree import DecisionTreeClassifier
nb =sklearn.tree.DecisionTreeClassifier(criterion='entropy',min_samples_split=5)
nb.fit(x_train, y_train)

test = nb.predict(x_test)
nb.score(x_test, y_test)
