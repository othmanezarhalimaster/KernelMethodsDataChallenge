# Kernel Methods Data challenge



# Package importation
import numpy as np
from math import exp,sqrt,tanh

import warnings
warnings.filterwarnings("ignore")



# Kernel Classes =======================================================================================================
class RBF:
    def __init__(self, sigma=1.):
        self.name = 'RBF'
        self.sigma = sigma  ## the variance of the kernel
        self.formula = lambda x, y: exp(-np.linalg.norm(x - y) ** 2 / (2 * self.sigma ** 2))

    def kernel_matrix(self, X, Y):
        ## Input: vectors X and Y of shape Nxd and Mxd
        ## Output: Matrix of shape NxM
        K=np.exp(np.array([[-np.linalg.norm(x-y)**2/(2*self.sigma**2) for x in X]for y in Y]))
        #return np.exp(-np.linalg.norm(X-Y[:,None],axis=-1)**2/(2*self.sigma**2))
        return K

class Sigmoid:
    def __init__(self, kappa=1.,intercept=1):
        self.name = 'Sigmoid'
        self.kappa = kappa  ## the variance of the kernel
        self.intercept = intercept
        self.formula = lambda x, y: tanh(self.kappa*x.T @ y+self.intercept)

    def kernel_matrix(self, X, Y):
        ## Input: vectors X and Y of shape Nxd and Mxd
        ## Output: Matrix of shape NxM
        return np.tanh(self.kappa*X @ (Y.T)+self.intercept)

class Linear:
    def __init__(self):
        self.name = 'Linear'
        self.formula = lambda x, y: x.T @ y

    def kernel_matrix(self, X, Y):
        ## Matrix of shape NxM
        # print('print kernel matrix')
        # print(X @ (Y.T))
        return X @ (Y.T)

class PolynomialKernel:
    def __init__(self,power,intercept):
        self.name = 'PolynomialKernel'
        self.intercept = intercept
        self.power = power
        self.formula = lambda x, y: (x.T @ y+self.intercept)**self.power

    def kernel_matrix(self, X, Y):
        ## Matrix of shape NxM
        return (X @ (Y.T)+self.intercept)**self.power

class MKL:
    def __init__(self, kernel_list, kernelweights):
        self.name = 'MKL'
        self.kernel_list = kernel_list
        if any(weight <= 0 for weight in kernelweights) == True:
            raise TypeError('MKL : at least one weight is non positive')
        else:
            self.kernelweights = kernelweights
            self.formula = lambda x, y: sum([weight * kernelobj.formula(x,y) for (weight, kernelobj) in zip(self.kernelweights, self.kernel_list)])

    def kernel_matrix(self, X, Y):
        ## Matrix of shape NxM
        return sum([weight * kernelobj.kernel_matrix(X, Y) for (weight, kernelobj) in zip(self.kernelweights, self.kernel_list)])

class GHI_Kernel:
    def __init__(self,beta):
        self.name = 'GHI_Kernel'
        self.beta = beta
        self.formula = lambda x, y: sum([min(abs(x[i])**self.beta,abs(y[i])**self.beta) for i in range(len(x))])

    def kernel_matrix(self, X, Y):
        ## Matrix of shape NxM
        return np.array([[np.sum(np.minimum(np.abs(x)**self.beta,np.abs(y)**self.beta)) for y in Y] for x in X])


class Chi_Kernel:
    def __init__(self,gamma):
        self.name = 'Chi_Kernel'
        self.gamma = gamma
        self.formula = lambda x, y: exp(-self.gamma*sum((x-y)**2/(x+y)))

    def kernel_matrix(self,X,y):
        out = np.zeros((X.shape[0], y.shape[0]))
        for i in range(X.shape[0]):
            for j in range(y.shape[0]):
                numerator = np.array(X[i])-np.array(y[j])
                denominator = np.array(X[i])+np.array(y[j])
                out[i, j] = sum(numerator**2/denominator)
        return np.exp(-self.gamma*out)
