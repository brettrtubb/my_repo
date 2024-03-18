
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

people_df = pd.read_csv(r"C:\Users\brett\source\repos\Baseball Pitcher Contract\Baseball Pitcher Contract\Stats\People.csv")
pitching_df = pd.read_csv(r"C:\Users\brett\source\repos\Baseball Pitcher Contract\Baseball Pitcher Contract\Stats\Pitching.csv")
position_df = pd.read_csv(r"C:\Users\brett\source\repos\Baseball Pitcher Contract\Baseball Pitcher Contract\Stats\Appearances.csv")
yearlystats_df = pd.read_csv(r"C:\Users\brett\source\repos\Baseball Pitcher Contract\Baseball Pitcher Contract\Stats\FanGraphs Leaderboard.csv")
teamstats_df = pd.read_csv(r"C:\Users\brett\source\repos\Baseball Pitcher Contract\Baseball Pitcher Contract\Stats\Teams.csv") 

teamstats_df.drop(columns = ['lgID', 'teamID', 'franchID', 'divID', 'Rank', 'Ghome', 'W', 'L', 'DivWin', 'WCWin', 'LgWin', 'WSWin', 'name', 'park', 'attendance', 'BPF', 'PPF', 'teamIDBR', 'teamIDlahman45', 'teamIDretro'], inplace = True)
year_df = teamstats_df.groupby('yearID')

people_df.drop(columns = ['birthMonth', 'birthDay', 'birthCountry', 'birthState', 'birthCity', 'deathYear', 'deathMonth', 'deathDay', 'deathCountry', 'deathState', 'deathCity', 'debut', 'finalGame', 'retroID', 'bbrefID', 'nameGiven', 'weight', 'height', 'bats', 'throws'], inplace = True)
people_df.dropna(inplace = True)
people_df['birthYear'] = people_df['birthYear'].astype(int)

pitching_df.drop(columns = ['stint', 'teamID', 'lgID', 'W', 'L', 'GS', 'CG', 'SHO', 'SV', 'H', 'ER', 'BAOpp', 'ERA', 'IBB', 'WP', 'BK', 'BFP', 'GF', 'R', 'SH', 'SF', 'GIDP'], inplace = True)
pitching_df = pitching_df.groupby(['playerID', 'yearID'], sort=False).sum().reset_index()
pitching_df = pitching_df.groupby('playerID').filter(lambda x : len(x) > 4)
pitching_df = pitching_df[pitching_df['G'] > 4]
pitching_df['HBP'] = pitching_df['HBP'].fillna(0)
pitching_df.dropna(inplace = True)
pitching_df = pitching_df[pitching_df.IPouts != 0]
pitching_df['IPouts'] = round((pitching_df['IPouts'] / 3), 1)
pitching_df.rename(columns = {'IPouts': 'IP'}, inplace = True)

pitching_df = pd.merge(pitching_df, people_df)
pitching_df['Age'] = pitching_df['yearID'] - pitching_df['birthYear']
pitching_df.drop(columns = ['birthYear', 'nameFirst', 'nameLast'], inplace = True)

yearlystats_df.rename(columns = {'Season': 'yearID'}, inplace = True)
yearlystats_df.drop(columns = ['wOBA', 'wOBAScale', 'wBB', 'wHBP', 'w1B', 'w2B', 'w3B', 'wHR', 'runSB', 'runCS', 'R/PA', 'R/W'], inplace = True)

pitching_df = pd.merge(pitching_df, yearlystats_df)

pitchers_df = pitching_df.groupby('playerID')

position_df['G_f'] = position_df['G_1b'] + position_df['G_2b'] + position_df['G_3b'] + position_df['G_of'] + position_df['G_dh'] + position_df['G_c'] + position_df['G_ss']
position_df.drop(columns=['yearID', 'teamID', 'lgID', 'G_all', 'GS', 'G_batting', 'G_1b', 'G_2b', 'G_3b', 'G_lf', 'G_of', 'G_dh', 'G_ph', 'G_defense', 'G_c', 'G_ss', 'G_cf', 'G_rf', 'G_pr'], inplace = True)
position_df = position_df.groupby('playerID')

groups = [2, 3, 4, 5, 6, 7, 9]
pit = pitchers_df.get_group('ryanno01').copy()
pit.columns
values = pit.values
i = 1
pyplot.figure()
for group in groups:
    pyplot.subplot(len(groups), 1, i)
    pyplot.plot(values[:, group])
    pyplot.title(pit.columns[group], y=0.5, loc='right')
    i += 1
pyplot.show()

pit.index = pit['Age']
pit.drop(columns = ['playerID', 'yearID'], inplace = True)
# pit

def pit_to_X_y(df, window_size = 10):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
        label = [r for r in df_as_np[i:i+window_size]]
        y.append(label)
    return np.array(X), np.array(y)
WINDOW_SIZE = 10
X,y = pit_to_X_y(pit, WINDOW_SIZE)
X.shape, y.shape

X_train, y_train = X[:14], y[:14]
X_test, y_test = X[14:], y[14:]
X_train.shape, y_train.shape, X_test.shape, y_test.shape

G_training_mean = np.mean(X_train[:,:,0])
G_training_std = np.std(X_train[:,:,0])

IP_training_mean = np.mean(X_train[:,:,1])
IP_training_std = np.std(X_train[:,:,1])

HR_training_mean = np.mean(X_train[:,:,2])
HR_training_std = np.std(X_train[:,:,2])

BB_training_mean = np.mean(X_train[:,:,3])
BB_training_std = np.std(X_train[:,:,3])

SO_training_mean = np.mean(X_train[:,:,4])
SO_training_std = np.std(X_train[:,:,4])

HBP_training_mean = np.mean(X_train[:,:,5])
HBP_training_std = np.std(X_train[:,:,5])

Age_training_mean = np.mean(X_train[:,:,6])
Age_training_std = np.std(X_train[:,:,6])

cFIP_training_mean = np.mean(X_train[:,:,7])
cFIP_training_std = np.std(X_train[:,:,7])

def preprocess(X):
    X[:,:,0] = (X[:,:,0] - G_training_mean)/G_training_std
    X[:,:,1] = (X[:,:,1] - IP_training_mean)/IP_training_std
    X[:,:,2] = (X[:,:,2] - HR_training_mean)/HR_training_std
    X[:,:,3] = (X[:,:,3] - BB_training_mean)/BB_training_std
    X[:,:,4] = (X[:,:,4] - SO_training_mean)/SO_training_std
    X[:,:,5] = (X[:,:,5] - HBP_training_mean)/HBP_training_std
    X[:,:,6] = (X[:,:,6] - Age_training_mean)/Age_training_std
    X[:,:,7] = (X[:,:,7] - cFIP_training_mean)/cFIP_training_std
    return X

def preprocess_output(y):
    y[:,:,0] = (y[:,:,0] - G_training_mean)/G_training_std
    y[:,:,1] = (y[:,:,1] - IP_training_mean)/IP_training_std
    y[:,:,2] = (y[:,:,2] - HR_training_mean)/HR_training_std
    y[:,:,3] = (y[:,:,3] - BB_training_mean)/BB_training_std
    y[:,:,4] = (y[:,:,4] - SO_training_mean)/SO_training_std
    y[:,:,5] = (y[:,:,5] - HBP_training_mean)/HBP_training_std
    y[:,:,6] = (y[:,:,6] - Age_training_mean)/Age_training_std
    y[:,:,7] = (y[:,:,7] - cFIP_training_mean)/cFIP_training_std
    return y

preprocess(X_train)
preprocess(X_test)
preprocess_output(y_train)
preprocess_output(y_test)

model = Sequential()
model.add(InputLayer((10,8)))
model.add(LSTM(150))
model.add(Dense(8))
model.compile(loss=MeanSquaredError(), optimizer='adam', metrics=[RootMeanSquaredError()])

history = model.fit(X_train, y_train, epochs=100, batch_size=72)

pyplot.plot(history.history['root_mean_squared_error'], label='RMSE')
pyplot.title('RSME for the player')
pyplot.ylabel('RSME value')
pyplot.xlabel('No. epoch')
pyplot.legend(loc='upper left')
pyplot.show()

def postprocessing_G(arr):
    arr = (arr*G_training_std) + G_training_mean
    return arr
def postprocessing_IP(arr):
    arr = (arr*IP_training_std) + IP_training_mean
    return arr
def postprocessing_HR(arr):
    arr = (arr*HR_training_std) + HR_training_mean
    return arr
def postprocessing_BB(arr):
    arr = (arr*BB_training_std) + BB_training_mean
    return arr
def postprocessing_SO(arr):
    arr = (arr*SO_training_std) + SO_training_mean
    return arr
def postprocessing_HBP(arr):
    arr = (arr*HBP_training_std) + HBP_training_mean
    return arr
def postprocessing_Age(arr):
    arr = (arr*Age_training_std) + Age_training_mean
    return arr
def postprocessing_cFIP(arr):
    arr = (arr*cFIP_training_std) + cFIP_training_mean
    return arr

predictions = model.predict(X_train)
actual = y_train
G_pred, IP_pred, HR_pred, BB_pred, SO_pred, HBP_pred, Age_pred, cFIP_pred = postprocessing_G(predictions[:, 0]), postprocessing_IP(predictions[:, 1]), postprocessing_HR(predictions[:, 2]), postprocessing_BB(predictions[:, 3]), postprocessing_SO(predictions[:, 4]), postprocessing_HBP(predictions[:, 5]), postprocessing_Age(predictions[:, 6]), postprocessing_cFIP(predictions[:, 7])
G_act, IP_act, HR_act, BB_act, SO_act, HBP_act, Age_act, cFIP_act = postprocessing_G(actual[:, 0]), postprocessing_IP(actual[:, 1]), postprocessing_HR(actual[:, 2]), postprocessing_BB(actual[:, 3]), postprocessing_SO(actual[:, 4]), postprocessing_HBP(actual[:, 5]), postprocessing_Age(actual[:, 6]), postprocessing_cFIP(actual[:, 7])

Predict = pd.DataFrame(data={'G': G_pred, 'G Actual': G_act, 'IP': IP_pred, 'IP Actual': IP_act, 'HR': HR_pred, 'HR Actual': HR_act, 'BB': BB_pred, 'BB Actual': BB_act,'SO': SO_pred, 'SO Actual': SO_act, 'HBP': HBP_pred, 'HBP Actual': HBP_act, 'Age': Age_pred, 'Age Actual': Age_act, 'cFIP': cFIP_pred, 'cFIP Actual': cFIP_act})
print(Predict)