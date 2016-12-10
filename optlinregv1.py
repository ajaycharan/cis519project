import csv
import numpy as np
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt
from sklearn import linear_model, svm, gaussian_process
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

stationToLOL = {}
stationToPos = {
    'Elm St': 0,
    'Norristown': 0, 
    'Main St': 1, 
    'Norristown TC': 2, 
    'Conshohocken': 3, 
    'Spring Mill': 4, 
    'Miquon': 5, 
    'Ivy Ridge': 6, 
    'Manayunk': 7, 
    'Wissahickon': 8, 
    'East Falls': 9, 
    'Allegheny': 10, 
    'North Broad St': 11, 
    'Temple U': 12, 
    'Jefferson Station': 13, 
    'Suburban Station': 14, 
    '30th Street Station': 15, 
    '30th St': 15,
    'University City': 16, 
    'Darby': 17, 
    'Curtis Park': 18, 
    'Sharon Hill': 19, 
    'Folcroft': 20, 
    'Glenolden': 21, 
    'Norwood': 22, 
    'Prospect Park': 23, 
    'Ridley Park': 24, 
    'Crum Lynne': 25, 
    'Eddystone': 26, 
    'Chester TC': 27, 
    'Highland Ave': 28, 
    'Marcus Hook': 29, 
    'Claymont': 30, 
    'Wilmington': 31, 
    'Churchmans Crossing': 32, 
    'Newark': 33
}
# TODO fill this in


statusLoc = 1
trainIdLoc = 0
dayLoc = 15
monthLoc = 17
# in secs
timestampdiffLoc = 13
sourceLoc = 7
nextLoc = 2
timeRawLoc = 16
lonLoc = 5
latLoc = 6

# End result below
# status, trainid, day, month, timestampdiff (secs), distance (calculated), time (converted to Unix)

def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def getDistance(sourceGPS, nextGPS):
    #Using Haversine formula
    sourceLon = float(sourceGPS[0])
    sourceLat = float(sourceGPS[1])
    nextLon = float(nextGPS[0])
    nextLat = float(nextGPS[1])
    sourceLon, sourceLat, nextLon, nextLat = map(radians, [sourceLon, sourceLat, nextLon, nextLat])

    dLon = nextLon - sourceLon
    dLat = nextLat - sourceLat
    a = sin(dLat/2)**2 + cos(sourceLat) * cos(nextLat) * sin(dLon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return int(c * r)


# preprocess the data 
with open('newark_norristown.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    # skip headers
    count = 0
    next(reader)
    for row in reader: 
        if '' in [row[statusLoc], row[trainIdLoc], row[dayLoc], row[monthLoc], row[timestampdiffLoc], row[sourceLoc], row[nextLoc], row[timeRawLoc]]: 
            continue
        if row[sourceLoc] in ['Bridge', 'None']:
            continue
        nextRow = [row[statusLoc], row[trainIdLoc], row[dayLoc], row[monthLoc], row[timestampdiffLoc]] 
        # calculate distance
        source = row[sourceLoc]
        sourceGPS = [row[lonLoc],row[latLoc]]
        
        nextStation = row[nextLoc]
        nextGPS = [row[lonLoc],row[latLoc]]

        distance = getDistance(sourceGPS, nextGPS)
        nextRow.append(distance)
        # convert to secs 
        secs = get_sec(row[timeRawLoc]) 
        nextRow.append(secs) 
        # add nextRow to appropriate matrix
        currentStation = -1
        if stationToPos[source] < stationToPos[nextStation]: 
            currentStation = stationToPos[nextStation] - 1
        else: 
            currentStation = stationToPos[nextStation] + 1
        # LEFT OFF HERE GET CURRENT STATIOn and dksjv,ajhdgjfkbs
        if (currentStation in stationToLOL.keys()): 
            stationToLOL[currentStation].append(nextRow)
        else: 
            stationToLOL[currentStation] = [nextRow]

stationScoreWithSplit = []
stationScoreCrossValidated = []
classifiersSplit = []
data_mat = []
station_name = []
for key, value in stationToLOL.iteritems():
    #data_mat.append(np.reshape(value, newshape=(len(value), '''7''' 6)))
    data_mat.append(np.reshape(value, newshape=(len(value), 7)))
    station_name.append(posToStation[key])

# data_mat has array for each matrix
score_mat = []
train_proportion = 0.7
num_iterations = 10
ind = 0
for stationData in data_mat:
    #stationData = data_mat[:,:,stationID]
    X = None 
    y = None
    X = stationData[:,1:]
    y = np.array([stationData[:, 0]]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_proportion, random_state=42)

    # n,d = X.shape
    # idx = np.arange(n)
    # np.random.seed(13)
    # np.random.shuffle(idx)
    # X = X[idx]
    # y = y[idx]
    # split_idx=train_proportion*n
    # X_train = X[:(split_idx - 1),:]
    # y_train = y[:(split_idx - 1),:]
    # X_test = X[split_idx:,:]
    # y_test = y[split_idx:,:]

    #rs = ShuffleSplit(n_splits=num_iterations, train_size=train_proportion, random_state=0)
    #clf = svm.SVR()
    clf = linear_model.LinearRegression()
    #clf = gaussian_process.GaussianProcessRegressor()
    clf = clf.fit(X_train, y_train)
    classifiersSplit.append(clf)
    y_pred = None
    y_pred = clf.predict(X_test) 
    #y_pred = abs(np.round(y_pred))
    y_pred[y_pred < 0] = 0
    y_pred = np.round(y_pred)
    #score = clf.score(X_test, y_test)
    #score = accuracy_score(y_test, y_pred)
    score = np.sum(abs(y_pred - y_test)) / y_pred.shape[0]
    stationScoreWithSplit.append(score)
    score_mat.append(score)
    # print 'for staion ID %d', ind
    # print 'the score for 70-30 simple split is: ', score
    
    # cross_scores = cross_val_score(clf, X, y,cv=5)
    # stationScoreCrossValidated.append(cross_scores)
    # print 'the cross validated scores are: ', cross_scores
    ind += 1 

results = [] 
results.append(score_mat) 
results.append(station_name)
resultFile = open("lrOutput.csv",'wb')
wr = csv.writer(resultFile, dialect='excel')
wr.writerows(results)
