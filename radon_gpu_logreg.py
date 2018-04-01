import numpy as np
import argparse
import os
import pycuda.autoinit
import pycuda.driver as drv
import numpy

from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from time import time
from copy import deepcopy
from pycuda.compiler import SourceModule

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--r',type=int,default=5) # useless, code decides
parser.add_argument('--h',type=int,default=2)
parser.add_argument('--datasize',type=int,default=10000)
parser.add_argument('--dataset',type=str,default='skin.csv')

parameters = parser.parse_args()
hyper = vars(parameters) #hyper -> dict containing all hyper parameters


# PyCUDA work here
mod = SourceModule("""
#define _CRT_SECURE_NO_WARNINGS
#include<stdlib.h>
#include<math.h>
#include<stdio.h>
#include<string.h>
#include<time.h>

#define ROW 4
#define TRAIN_SIZE 20000
#define TEST_SIZE 4000
#define VERBOSE 0
#define L2 0.1
#define TOL 0.0001

float sig(float *w, float *x, int m) {
	float s = 0;
	for (int i = 0; i < m; i++) {
		s -= w[i] * x[i];
	}
	return 1 / (1 + expf(s));
}

float* error(float *X, float *Y, float *W, float *err, int m, int n) {
	float *temp = (float*)calloc(m, sizeof(float));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			temp[j] = X[m*i + j];
		}
		err[i] = Y[i] - sig(W, temp, m);
	}
	free(temp);
	return err;
}

float MSE(float *X, float *Y, float *W, float *err, int m, int n) {
	float *temp = (float*)calloc(m, sizeof(float));
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			temp[j] = X[m*i + j];
		}
		err[i] = Y[i] - sig(W, temp, m);
	}
	free(temp);
	float mse = 0;
	for (int i = 0; i < n; i++){
		mse += err[i] * err[i];
	}
	return mse/n;
}

float* gradient(float *X, float *Y, float *W, float *grad, float *err, int m, int n) {
	err = error(X, Y, W, err, m, n);

	float val = 0;

	for (int i = 0; i < n; i++) {
		val += X[i*m] * err[i];
	}
	grad[0] = 0;

	for (int j = 1; j < m; j++) {
		val = 0;
		for (int i = 0; i < n; i++) {
			val += X[i*m + j] * err[i];
		}
		grad[j] = val - L2*W[j];
	}
	return grad;
}

float* train(int m, int n, float *X, float *Y, float *W, float lr, int n_epochs) {
	float *grad = (float*)calloc(m, sizeof(float));
	float *err = (float*)calloc(n, sizeof(float));
	float temp = 1000.0;
	float mse = 1000.0;
	int steps = 0;

	int train_size = 0.9 * n;
	float *X_train = (float*)calloc(train_size*m, sizeof(float));
	float *Y_train = (float*)calloc(train_size, sizeof(float));
	float *X_val = (float*)calloc((n-train_size)*m, sizeof(float));
	float *Y_val = (float*)calloc((n-train_size), sizeof(float));

	for (int i=0; i<train_size; i++){
		X_train[i] = X[i];
		Y_train[i] = Y[i];
	}
	for (int i=0; i<n-train_size; i++){
		X_val[i] = X[i+train_size];
		Y_val[i] = Y[i+train_size];
	}

	W[0] = 1;
	while (mse > TOL && steps < n_epochs) {
		steps ++;
		grad = gradient(X_train, Y_train, W, grad, err, m, train_size);
		for (int j = 0; j < m; j++) {
			W[j] += lr * grad[j];
		}
		temp = MSE(X_val, Y_val, W, err, m, n-train_size);
		if (temp > mse){
			steps = n_epochs;
		}
		else{
			mse = temp;
		}
	}

	free(grad);
	free(err);
	free(X_train);
	free(Y_train);
	free(X_val);
	free(Y_val);
	return W;
}


__global__ void trainer(float *dest, float *a, float *b){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;


    //dest[i] = a[i] * b[i] * b[i] * a[i];
}

""")

trainer = mod.get_function("trainer")

# exit('Safe, Gentlemen!')

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

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
Y_train = Y_train.astype(np.float32)
Y_test = Y_test.astype(np.float32)

#invoke the model
model = LogisticRegression()


#training
start = time()
r = X_train.shape[1] + 3 #hyper['r']
h = hyper['h']
S = [] # hypotheses

# synthesis
for _ in range(r**h):
    x_train,y_train = shuffle(X_train,Y_train,n_samples=hyper['datasize'])
    model.fit(x_train,y_train)
    # print(model.coef_,model.intercept_)
    # exit()
    S.append(list(model.coef_[0]) + list(model.intercept_)) # storing hypotheses
S = np.array(S)

# aggregation
for _ in range(h,0,-1):
    S_new = []
    for part in batches(S,r):
        r_pt = radon_point(part)
        S_new.append(r_pt)
    S = np.array(S_new)
    np.random.shuffle(S) # to make it iid like

# print('Finished with aggregation')
# print(S)
# exit()
weights = np.array([S[0,:-1]])
bias = np.array([S[0,-1]])
model.coef_ = weights
model.intercept_ = bias

end = time()


#evaluating
y_preds = model.predict(X_test)
fscore = f1_score(Y_test,y_preds,average='micro')

print('Test Performance: {0:.2f}%'.format(100*fscore))
print('Train Time: {0:.2f} seconds'.format(end-start))
