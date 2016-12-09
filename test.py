import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import cross_val_score
    
d0 = 5000
d1 = 7
d2 = 50
stationScoreWithSplit = {}
stationScoreCrossValidated = {}
classifiersSplit = {}
data_mat = np.random.rand(d0, d1, d2)
train_proportion = 0.7
num_iterations = 10
for stationID in range(data_mat.shape[2]):
    stationData = data_mat[:,:,stationID]
    X = stationData[:,1:]
    y = np.array([stationData[:, 0]])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_proportion, random_state=42)
    #rs = ShuffleSplit(n_splits=num_iterations, train_size=train_proportion, random_state=0)
    clf = linear_model.LinearRegression()
    clf.fit(X_train, y_train)
    classifiersSplit.append(clf)
    score = clf.score(X_test, y_test)
    stationScoreWithSplit.append(score)
    print 'for staion ID %d', stationID
    print 'the score for 70-30 simple split is: ', score
    
    cross_scores = cross_val_score(clf, X, y,cv=5)
    stationScoreCrossValidated.append(cross_scores)
    print 'the cross validated scores are: ', cross_scores
