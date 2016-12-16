import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, svm, gaussian_process, ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import argparse
import sys
import datetime

# this is where the temp matrices are stored
stationToLOL = {}
# TODO: change in each file 
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

# TODO: make sure this part is right
posToStation = {} 
for key, value in stationToPos.iteritems(): 
    posToStation[value] = key
posToStation[0] = 'Norristown/Elm St'
posToStation[15] = '30th Street Station'


statusLoc = 1
trainIdLoc = 0
dayLoc = 15
monthLoc = 17
# in secs
timestampdiffLoc = 13
sourceLoc = 7
nextLoc = 2
timeRawLoc = 16

# End result below
# status, day, month, timestampdiff (secs), distance (calculated), time (converted to Unix)

def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

# preprocess the data 
# TODO: make sure this part is correct 
with open('newark_norristown.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile)
    # skip headers
    count = 0
    next(reader)
    for row in reader: 
        if '' in [row[statusLoc], row[dayLoc], row[monthLoc], row[timestampdiffLoc], row[sourceLoc], row[nextLoc], row[timeRawLoc]]: 
            continue
        # TODO: this is specific
        if row[sourceLoc] in ['Bridge', 'None']:
            continue
        nextRow = [int(row[statusLoc]), int(row[dayLoc]), int(row[monthLoc]), int(row[timestampdiffLoc])] 
        # calculate distance
        source = row[sourceLoc]
        nextStation = row[nextLoc] 
        distance = abs(stationToPos[nextStation] - stationToPos[source]) - 1
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
    data_mat.append(np.reshape(value, newshape=(len(value), 6)))
    station_name.append(posToStation[key])

# data_mat has array for each matrix
score_mat = []
score_RF_mat = []
avg_delay_error_RF_mat = []
train_proportion = 0.7
num_iterations = 10
for stationData in data_mat:
    X = None 
    y = None
    X = stationData[:,1:]
    y = np.array([stationData[:, 0]]).T
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_proportion, random_state=42)
    # TODO: type of regression here
    clf = linear_model.LinearRegression()
    clf = clf.fit(X_train, y_train)
    classifiersSplit.append(clf)
    y_pred = None
    y_pred = clf.predict(X_test) 
    y_pred[y_pred < 0] = 0
    y_pred = np.round(y_pred)
    score = np.sum(abs(y_pred - y_test)) / y_pred.shape[0]
    #stationScoreWithSplit.append(score)
    score_mat.append(score)
    # Random Forest:
    clf2 = ensemble.RandomForestRegressor(n_estimators=10, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-07, bootstrap=True)
    clf2 = clf2.fit(X_train, np.ravel(y_train))
    y_pred_RF = None
    y_pred_RF = clf2.predict(X_test)
    y_pred_RF[y_pred_RF < 0] = 0
    y_pred_RF = np.round(y_pred_RF)
    '''
    print "-------------------------"
    print "Raw Prediction: ", y_pred_RF
    print "-------------------------"
    '''
    score_RF = clf2.score(X_test, y_test)
    score_RF_mat.append(score_RF)
    avg_delay_error_RF = None
    avg_delay_error_RF = np.sum(abs(y_pred_RF - y_test.T)) / len(y_pred_RF)
    avg_delay_error_RF_mat.append(avg_delay_error_RF)
    
    print "-------------------------"
    print "Random Forest Score: ", score_RF
    print "-------------------------"
    
results = [] 
results.append(score_mat) 
results.append(station_name)
# TODO: make this part specific 
resultFile = open("lrOutput.csv",'wb')
wr = csv.writer(resultFile, dialect='excel')
wr.writerows(results)

results_RF = []
results_RF.append(avg_delay_error_RF_mat) 
results_RF.append(station_name)
resultFile_RF = open("rfOutput.csv",'wb')
wr_RF = csv.writer(resultFile_RF, dialect='excel')
wr_RF.writerows(results_RF)

while(1):
    '''
    parser = argparse.ArgumentParser(description='Please enter the day of the week when you would like to know the status: ')
    parser.add_argument("day", help="enter the day of the week", type=int)
    '''
    
    source_in = raw_input("Please enter the source station: ")
    nextStation_in = raw_input("Please enter the station at which you would like  to know the status: ")
    now = raw_input("Do you want to know the status at current time or later? (y/n): ")
    if(now=='n'):
        day_in = raw_input("Please enter the day of the week when you would like to know the status: ")
        month_in = raw_input("Please enter the month (from March to May) when you would like to know the status (3-5): ")
        time_raw = raw_input("Please enter the time when you would like to know the status(24hr format): ")
        time_in = (int(time_raw[0])*10 + int(time_raw[1]))*3600 + int(time_raw[2])*10 + int(time_raw[3])*60
        print"the human time in sec is", time_in
    else:
        d = datetime.datetime.now()
        day_in = d.isoweekday()
        month_in = d.month
        hour_in = d.hour
        minute_in = d.minute
        second_in = d.second
        time_in = int(hour_in) * 3600 + int(minute_in) * 60 + int(second_in)
        print"the machine time in sec is", time_in

    quit = raw_input("Do you want to quit?(y/n): ")
    if(quit=='y'):
        print('Thanks for your time :)')
        break
    else:
        continue
