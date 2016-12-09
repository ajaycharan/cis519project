import csv
import numpy as np

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

# End result below
# status, trainid, day, month, timestampdiff (secs), distance (calculated), time (converted to Unix)

def get_sec(time_str):
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

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
        # LEFT OFF HERE GET CURRENT STATIOn and dksjv,ajhdgjfkbs
        if (currentStation in stationToLOL.keys()): 
            stationToLOL[currentStation].append(nextRow)
        else: 
            stationToLOL[currentStation] = [nextRow]

arrOfMatrices = []
for key, value in stationToLOL.iteritems():
    arrOfMatrices.append(np.reshape(value, newshape=(len(value), 7)))

# arr OfMatrices has array for each matrix
