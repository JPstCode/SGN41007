from scipy.io import loadmat
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression

data = loadmat('arcene.mat')

x_train = data['X_train']
x_test = data['X_test']
y_train = data['y_train'].ravel()
y_test = data['y_test'].ravel()

clf = LogisticRegression(random_state=0, solver='lbfgs')
rfe = RFECV(clf, step=50, verbose=1)

rfe = rfe.fit(x_train,y_train)

print(rfe.support_)

plt.plot(range(0,10001,50), rfe.grid_scores_)


predict = rfe.predict(x_test)

print(accuracy_score(y_test,predict))
plt.show()