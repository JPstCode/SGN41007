import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
    AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


f = np.load(r'P:\My Documents\PycharmProjects\SGN41007\Ex4\GTSRB_subset\gtsrb_f.npy')
y = np.load(r'P:\My Documents\PycharmProjects\SGN41007\Ex4\GTSRB_subset\gtsrb_y.npy')

rf = RandomForestClassifier(n_estimators=100)
etf = ExtraTreesClassifier(n_estimators=100)
ada = AdaBoostClassifier(n_estimators=100)
gbc = GradientBoostingClassifier(n_estimators=100)


xtrain, xtest, ytrain, ytest = train_test_split(f,y, test_size= 0.2)

#a)

rf.fit(xtrain,ytrain)
predict1 = rf.predict(xtest)

print('Random Forest: ', accuracy_score(predict1,ytest))


#b)

etf.fit(xtrain,ytrain)
predict2 = etf.predict(xtest)

print('Extremely Randomised Trees:', accuracy_score(predict2,ytest))

#c)

ada.fit(xtrain,ytrain)
predict3 = ada.predict(xtest)

print('Adaboost:', accuracy_score(predict3,ytest))

#d)

gbc.fit(xtrain,ytrain)
predict4 = gbc.predict(xtest)

print('Gradient boosted Tree:', accuracy_score(predict4,ytest))