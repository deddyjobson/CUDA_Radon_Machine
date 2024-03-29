import numpy as np
import argparse
import os
import pycuda.autoinit
import pycuda.driver as cuda
import numpy

from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.utils import shuffle, resample
from time import time, sleep
from copy import deepcopy
from pycuda.compiler import SourceModule
from sklearn.decomposition import PCA

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--r',type=int,default=-1) # default depends on X_shape
parser.add_argument('--h',type=int,default=2)
parser.add_argument('--datasize',type=int,default=1000) 
parser.add_argument('--dataset',type=str,default='skin.csv')

parameters = parser.parse_args()
hyper = vars(parameters) #hyper -> dict containing all hyper parameters


# PyCUDA work here
mod = SourceModule("""
#define _CRT_SECURE_NO_WARNINGS

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdlib.h>
#include<math.h>


__device__ float sig(float *w, float *x, int m) {
    float s = 0;
    for (int i = 0; i < m; i++) {
        s -= w[i] * x[i];
    }
    return 1 / (1 + expf(s));
}

__device__ float MSE(float *X, float *Y, float *W, int m, int n) {
    float temp = 0;
    float mse = 0;
    for (int i = 0; i < n; i++) {
        temp = Y[i] - sig(W, X+m*i, m);
        mse += temp * temp;
    }
    return mse/n;
}


__device__ float* gradient(float *X, float *Y, float *W, float *grad, int m, int n) {
    float val = 0;

    grad[0] = 0;

    for (int i = 0; i < n; i++) {
        val = Y[i] - sig(W, X+m*i, m);
        for (int j = 1; j < m; j++) {
            grad[j] += X[i*m + j] * val;
        }
    }
    for (int j = 1; j < m; j++) {
        grad[j] -= W[j];
    }

    return grad;
}

__device__ void train(int m, int n, float *X, float *Y, float *W, float *grad, float *temp_weights, float lr, int n_epochs) {
    float temp = 1000.0;
    float mse = 1000.0;
    int steps = 0;
    int train_size = int(0.9 * n);

    int terminate = -1;
    W[0] = 1; // set bias fixed to 1
    while (mse > 0.0001 && steps < n_epochs) {
        terminate++;
        steps ++;
        grad = gradient(X, Y, W, grad, m, train_size);
        for (int j = 0; j < m; j++) {
            W[j] += lr * grad[j]; // we ascend the gradient here.
        }
        temp = MSE(X+m*train_size, Y+train_size, W, m, n-train_size);

        if(temp < mse){
            terminate = -1;
            mse = temp;
            for (int i=0; i < m; i++){ // save model
                temp_weights[i] = W[i];
            }
        }
        if (terminate >= 10){ // early stopping
            steps = n_epochs;
            for (int i=0; i < m; i++){ // restore best model
                W[i] = temp_weights[i];
            }
        }

    }

}


__global__ void trainer(float *W, int m, int n, int data_size, float *X, float *Y, float *grad, float *temp_weights){
    const int tid = blockIdx.y * gridDim.x + blockIdx.x;
    //const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    train(m, n, X + n*m*tid, Y + n*tid, W + tid*m, grad+tid*m, temp_weights+tid*m, 0.001, 20000);
}
""")
# printf("Blah--%d--%d--%d...\\n",n*m*tid + i*m + j,m*data_size,tid);

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
    lim = -1 # 10**6
    # X = data[:lim,1:22]
    X = data[:lim,-7:]
    y = data[:lim,0]
elif hyper['dataset'] == 'SUSY.csv':
    X = data[:,1:]
    y = data[:,0]
elif hyper['dataset'] == 'HEPMASS_wide.csv':
    lim = -1 # 10**6
    # X = data[:lim,1:22]
    X = data[:lim,1:]
    y = data[:lim,0]
elif hyper['dataset'] == 'HIGGS.csv':
    # X = data[:,1:22]
    X = data[:,-7:]
    y = data[:,0]
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
    

    X_train = np.hstack((np.ones_like(Y_train).reshape(-1,1),X_train))
    X_test = np.hstack((np.ones_like(Y_test).reshape(-1,1),X_test))

    # del data,X,y # not used anymore

    # print(list(map(lambda d:d.shape,(X_train, X_test, Y_train, Y_test))))

    #pre-process the data
    prep = StandardScaler(copy=False)
    prep.fit_transform(X_train)
    prep.transform(X_test)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    Y_test = Y_test.astype(np.float32)


    #training
    start = time()
    if hyper['r'] <= 0:
        r = X_train.shape[1] + 2 # (#dims in hyperspace) +2
    else:
        r = hyper['r']
    h = hyper['h']

    # synthesis - GPU version
    # X_train,Y_train = shuffle(X_train,Y_train)
    X_train,Y_train = resample(X_train, Y_train, replace=True, n_samples=hyper['datasize']*(r**h))
    S = np.zeros(r**h * X_train.shape[1]).astype(np.float32) 

    # trainer(float *dest, int m, int n, int data_size, float *X, float *Y)
    X_shape = X_train.shape
    X_train = X_train.reshape(-1)
    Y_train = Y_train.reshape(-1)

    # print(X_shape,np.int32(r-2),np.int32(X_shape[0]//(r**h)))
    # exit()

    S_gpu = cuda.mem_alloc(S.nbytes)
    cuda.memcpy_htod(S_gpu, S)
    X_train_gpu = cuda.mem_alloc(X_train.nbytes)
    cuda.memcpy_htod(X_train_gpu, X_train)
    Y_train_gpu = cuda.mem_alloc(Y_train.nbytes)
    cuda.memcpy_htod(Y_train_gpu, Y_train)
    grad = cuda.mem_alloc(np.zeros(X_shape[1], dtype=np.float32).nbytes)
    temp_weights = cuda.mem_alloc(np.zeros(X_shape[1], dtype=np.float32).nbytes)


    trainer(
            S_gpu, np.int32(X_shape[1]), np.int32(X_shape[0]//(r**h)), np.int32(X_shape[0]),
             X_train_gpu, Y_train_gpu, grad, temp_weights,
            block=(1,1,1), grid=(r**(h-1),r)) # no sharing of data so do it in separate cores
            # block=(r,1,1), grid=(r**(h-1),1)) # no sharing of data so do it in separate cores

    X_train_gpu.free()
    Y_train_gpu.free()
    grad.free()
    temp_weights.free()

    cuda.memcpy_dtoh(S, S_gpu)
    S_gpu.free()

    S = S.reshape((r**h , X_shape[1]))
    # S = S[:,1:]
    # print(S.shape)
    # exit()

    S += np.random.randn(*S.shape)*1e-5 # to ensure no two exactly equal hypotheses

    if r < X_shape[1] + 2: # then we must apply dimension reduction to r
        dim_red = PCA(n_components=r-2)
        S = dim_red.fit_transform(S)
        # print(S.shape)
        # exit()


    weights = 1.0 / S.shape[0] * np.sum(S,axis=0)

    end = time()

    def predict(x,w=weights):
        return (np.sum(w * x, axis=1) > 0).astype(np.int32)

    #evaluating
    y_preds = predict(X_test, weights)

    fscore = f1_score(Y_test,y_preds,average='binary')

    fscores.append(fscore*100)
    times.append(end-start)

net_fscore = sum(fscores) / len(fscores)
net_time = sum(times) / len(times)

# print(fscores, times)
print('Test Performance: {0:.2f}%'.format(net_fscore))
print('Train Time: {0:.4f} seconds'.format(net_time))
