import numpy as np
import argparse
import os

from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from time import time


np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--r',type=int,default=5)
parser.add_argument('--h',type=int,default=2)
parser.add_argument('--dataset',type=str,default='skin.csv')

parameters = parser.parse_args()
hyper = vars(parameters) #hyper -> dict containing all hyper parameters

#global variables here


#getting data
data = read_csv(os.path.join('..','Datasets',hyper['dataset'])).as_matrix()
# print(data.shape)
# exit()

if hyper['dataset'] == 'skin.csv':
    X = data[:,:-1]
    y = data[:,-1]
elif hyper['dataset'] == 'fashion_MNIST.csv':
    X = data[:,1:-1]
    y = data[:,-1]
    indices = (y <= 1)
    X = X[indices]
    y = y[indices]
elif hyper['dataset'] == 'HEPMASS.csv':
    lim = -1 #10**6
    # X = data[:lim,1:22]
    X = data[:lim,-7:]
    y = data[:lim,0]
elif hyper['dataset'] == 'SUSY.csv':
    X = data[:,-10:]
    y = data[:,0]
elif hyper['dataset'] == 'HIGGS.csv':
    # X = data[:,1:22]
    X = data[:,-7:]
    y = data[:,0]
elif hyper['dataset'] == 'HEPMASS_wide.csv':
    lim = -1 # 10**6
    # X = data[:lim,1:22]
    X = data[:lim,1:]
    y = data[:lim,0]
elif hyper['dataset'] == 'HIGGS_wide.csv':
    lim = -1 # 10**6
    # X = data[:lim,1:22]
    X = data[:lim,1:]
    y = data[:lim,0]
elif hyper['dataset'] == 'gen1-2-100000000.csv':
    X = data[:,-2:]
    y = data[:,0]
else:
    print('Dataset doesn\'t exist')

del data 

skf = StratifiedKFold(n_splits=10)

times = [] # list of the train time for each split
fscores = [] # list of the train fscore for each split

for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    

    #pre-process the data
    prep = StandardScaler(copy=False)
    prep.fit_transform(X_train)
    prep.transform(X_test)

    #invoke the model
    model = LogisticRegression()

    #training
    start = time()
    model.fit(X_train,Y_train)
    end = time()

    #evaluating
    y_preds = model.predict(X_test)
    fscore = f1_score(Y_test,y_preds,average='binary')

    fscores.append(fscore*100)
    times.append(end-start)

net_fscore = sum(fscores) / len(fscores)
net_time = sum(times) / len(times)

# print(fscores, times)
print('Test Performance: {0:.2f}%'.format(net_fscore))
print('Train Time: {0:.4f} seconds'.format(net_time))
