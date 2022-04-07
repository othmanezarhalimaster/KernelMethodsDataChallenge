# Kernel Methods Data challenge



# Package importation
import numpy as np
import pandas as pd
from math import exp,sqrt,tanh
from scipy import optimize

import warnings
warnings.filterwarnings("ignore")

from numpy import linalg as LA




# Model trainer ========================================================================================================
class KernelMSVC:

    def __init__(self, C, kernel, epsilon=1e-8):
        self.type = 'non-linear'
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
        # For prediction
        self.dataset = None
        self.size_dataset = None
        self.Numclasses = None
        self.predictions = None

    def fit(self, X, y):
        self.size_dataset = len(y)
        self.Numclasses = len(set(y))
        self.dataset = X
        K = self.kernel.kernel_matrix(X, X)
        # Masashi SugiyamaMasashi Sugiyama, in Introduction to Statistical Machine Learning, 2016 ======================
        # https://www.sciencedirect.com/topics/computer-science/multiclass-classification
        global_K = np.block([[K for i in range(self.Numclasses)]for i in range(self.Numclasses)])

        print('global kernel matrix conditioning = ',LA.cond(global_K))
        # Lagrange dual problem
        def loss(alpha):
            # '''--------------dual loss ------------------ '''
            ycheck = []
            for i in range(0, self.size_dataset * self.Numclasses, self.size_dataset):
                for x in alpha[i:self.size_dataset + i]:
                    if y[list(alpha[i:self.size_dataset + i]).index(x)] != i/self.size_dataset:
                        ycheck.append(1)
                    else:
                        ycheck.append(0)
            ycheck = np.array(ycheck)
            return 1 / 2 * (alpha).T @ (global_K) @ (alpha)-sum(alpha*ycheck)
        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            # '''----------------partial derivative of the dual loss wrt alpha-----------------'''
            gradsum = []
            for i in range(0, self.size_dataset * self.Numclasses, self.size_dataset):
                for x in alpha[i:self.size_dataset+i]:
                    if y[list(alpha[i:self.size_dataset + i]).index(x)] != i/self.size_dataset:
                        gradsum.append(1)
                    else:
                        gradsum.append(0)
            wholegrad = (global_K) @ (alpha)-gradsum
            return wholegrad

        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0
        # '''----------------function defining the equality constraint------------------'''  #
        def fun_eq(alpha):
            return sum(alpha[i:self.size_dataset + i] for i in range(0, self.size_dataset * self.Numclasses, self.size_dataset))

        def fun_ineq(alpha):
            funineqlist =[]
            for i in range(0, self.size_dataset * self.Numclasses, self.size_dataset):
                for x in alpha[i:self.size_dataset + i]:
                    if y[list(alpha[i:self.size_dataset + i]).index(x)] != i/self.size_dataset:
                        funineqlist.append(-x)
                    else:
                        funineqlist.append(self.C -x)
            return np.array(funineqlist)

        constraints = [{'type': 'ineq','fun':fun_ineq}] + [{'type': 'eq', 'fun': fun_eq}]
        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=0.5*np.ones(self.size_dataset* self.Numclasses),
                                   method='SLSQP',
                                   jac=lambda alpha: grad_loss(alpha), constraints=constraints,
                                   options={'maxiter': 50})

        self.alpha = optRes.x
        print(optRes)


    def separating_function_Singleobs(self, x):
        # Input : matrix x of shape N data points times d dimension
        # Output: scalar in {0,1,2,...,C}
        alphas = [self.alpha[i:self.size_dataset + i] for i in range(0, self.size_dataset * self.Numclasses, self.size_dataset)]
        probaofclasses =[sum(alpha * np.array([self.kernel.formula(x_i, x) for x_i in self.dataset])) for alpha in alphas]
        return np.argmax(probaofclasses)

    def predict(self, X_test):
        """ Predict y values in {1,2,3,...,C} """
        return [self.separating_function_Singleobs(x) for x in X_test]

    def l0Accuracy(self,X_test,ytarget):
        test_list = (self.predict(X_test)==ytarget)
        return len(test_list[test_list==True])/len(ytarget)

    def Submission(self,X_test):
        Submission_path = "D://Master-2021-2022//MASH//Courses//Optional courses//Kernel methods//Data challenge//Submission.csv"
        Predictions = self.predict(X_test)
        predicted_items_dict = {'Id':range(1,len(Predictions)+1),'Prediction':Predictions}
        dataframe_predicted_items_dict=   pd.DataFrame(predicted_items_dict)
        dataframe_predicted_items_dict.to_csv(Submission_path,index=False)
        return "Submission file construction succeeded"


