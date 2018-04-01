import numpy as np
import argparse
import os

from pandas import read_csv
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from time import time
from copy import deepcopy
from operator import mul
from functools import reduce


np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--r',type=int,default=5) # useless; code decides
parser.add_argument('--h',type=int,default=2)
parser.add_argument('--datasize',type=int,default=10000)
parser.add_argument('--dataset',type=str,default='skin.csv')

parameters = parser.parse_args()
hyper = vars(parameters) #hyper -> dict containing all hyper parameters


#functions here
def batches(lst, bs):# splits data into batches
    for i in range(0,len(lst),bs):
        yield lst[i:i+bs]

def radon_point(pts):
    A = np.hstack((pts[:-1],np.ones((pts.shape[0]-1,1))))
    b = np.concatenate((pts[-1],np.ones(1)))
    alphas = np.linalg.solve(A.T,b)
    alphas = np.concatenate((alphas,[-1]))
    alpha_plus = alphas * (alphas>0)
    # print('alphas:',alphas)
    # print(np.sum(alpha_plus[:,np.newaxis] * pts, axis=0) / np.sum(alpha_plus))
    return np.sum(alpha_plus[:,np.newaxis] * pts, axis=0) / np.sum(alpha_plus)

class Shape:
    def __init__(self,model):
        self.model = model
        self.shapew = None
        self.shapeb = None
    def flatten(self):
        w = self.model.coefs_
        b = self.model.intercepts_
        self.shapew = [x.shape for x in w]
        self.shapeb = [x.shape for x in b]
        pt = []
        for l in w:
            pt.extend(l.reshape(-1))
        for l in b:
            pt.extend(l.reshape(-1))
        return pt
    def unflatten(self,pt):
        w = []
        b = []
        i = 0
        for shape in self.shapew:
            size = reduce(mul, shape, 1)
            w.append(np.array(pt[i:i+size]).reshape(shape))
            i += size
        for shape in self.shapeb:
            size = reduce(mul, shape, 1)
            b.append(np.array(pt[i:i+size]).reshape(shape))
            i += size
        return w,b

#getting data
data = read_csv(os.path.join('Datasets',hyper['dataset'])).as_matrix()
# print(data.shape)
# exit()

if hyper['dataset'] == 'skin.csv':
    X = data[:,:-1]
    y = data[:,-1]
elif hyper['dataset'] == 'HEPMASS.csv':
    X = data[:,1:]
    y = data[:,0]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, stratify=y, test_size=0.2)
del data,X,y # not used anymore

# print(list(map(lambda d:d.shape,(X_train, X_test, Y_train, Y_test))))

#pre-process the data
prep = StandardScaler(copy=False)
prep.fit_transform(X_train)
prep.transform(X_test)

#invoke the model
model = MLPClassifier(hidden_layer_sizes = (10,))
flat = Shape(model)

model.fit([X_train[0]],[Y_train[0]])
r = len(flat.flatten()) + 2 #hyper['r']
h = hyper['h']
S = [] # hypotheses

#training
start = time()

# synthesis
for _ in range(r**h):
    x_train,y_train = shuffle(X_train,Y_train,n_samples=hyper['datasize'])
    model.fit(x_train,y_train)
    S.append(flat.flatten()) # storing hypotheses
S = np.array(S)

end0 = time()
# aggregation
for _ in range(h,0,-1):
    S_new = []
    for part in batches(S,r):
        r_pt = radon_point(part)
        S_new.append(r_pt)
    S = np.array(S_new)
    np.random.shuffle(S) # to make it iid-like

# print('Finished with aggregation')
# print(S)
# exit()

model.coefs_, model.intercepts_ = flat.unflatten(S[0])

end = time()


#evaluating
y_preds = model.predict(X_test)
fscore = f1_score(Y_test,y_preds,average='micro')

print('Test Performance: {0:.2f}%'.format(100*fscore))
print('Synthesis Time: {0:.2f} seconds'.format(end0-start))
print('Agregation Time: {0:.2f} seconds'.format(end-end0))
print('Total Time: {0:.2f} seconds'.format(end-start))
