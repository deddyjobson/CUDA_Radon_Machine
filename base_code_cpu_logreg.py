import numpy as np
import argparse
import os

from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
data = read_csv(os.path.join('Datasets',hyper['dataset'])).as_matrix()
# print(data.shape)
# exit()

if hyper['dataset'] == 'skin.csv':
    X = data[:,:-1]
    y = data[:,-1]
elif hyper['dataset'] == 'HEPMASS.csv':
    lim = 10**6
    X = data[:lim,1:]
    y = data[:lim,0]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y, test_size=0.2)
del data,X,y # not used anymore

# print(list(map(lambda d:d.shape,(X_train, X_test, Y_train, Y_test))))

#pre-process the data
prep = StandardScaler(copy=False)
prep.fit_transform(X_train)
prep.transform(X_test)

#invoke the model
model = LogisticRegression(fit_intercept = False, n_jobs=1, solver='liblinear')

#training
start = time()
model.fit(X_train,Y_train)
end = time()

#evaluating
y_preds = model.predict(X_test)
fscore = f1_score(Y_test,y_preds,average='binary')

print('Test Performance: {0:.2f}%'.format(100*fscore))
print('Train Time: {0:.2f} seconds'.format(end-start))
