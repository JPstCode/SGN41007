from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt

#a)
mat = loadmat('EX1/twoClassData.mat')

xmat = mat['X']
ymat = mat['y']


#b)

tr = xmat[:200]
test = xmat[200:]

#plt.figure(0)
#plt.plot(tr[:,0], tr[:,1], 'ro')
#plt.plot(test[:,0], test[:,1], 'bo')

#plt.show()

#c)

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(xmat,ymat[0])

knn_predicted = knn.predict(test)
knn_score = accuracy_score(y_true=ymat[0][200:],
                           y_pred=knn_predicted)

print("knn acc-score:", knn_score)

#d)

lda = LinearDiscriminantAnalysis()
lda.fit(xmat,ymat[0])
lda_predicted = lda.predict(test)

lda_score = accuracy_score(y_true=ymat[0][200:],
                           y_pred=lda_predicted)

print("lda acc-score:", lda_score)
