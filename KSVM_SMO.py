# Kernel Methods Data challenge



import numpy as np
import random
from tqdm import tqdm
random.seed(123)
import pandas as pd

class KernelSVMBinaryClassifier:
    """
    X: support vectors
    alpha: corresponding coefficients for predictions
    kernel: SVM's kernel
    class1, class2: original labels of the classes
    """
    def __init__(self, kernel):
        self.kernel = kernel
        self.X = None
        self.alpha = None
        self.bias = None
        self.class1 = None
        self.class2 = None

    def Dual_objective(self, i, j, ai, K, y, alpha):
        aj = alpha[i] + alpha[j] - ai
        ret = 2 * ai * y[i] + 2 * aj * y[j]
        ret -= ai * K[i, i] * ai + aj * K[j, j] * aj + 2 * ai * K[i, j] * aj

        n = K.shape[0]
        for k in range(n):
            if k != i and k != j:
                ret -= 2 * (ai * K[i, k] + aj * K[j, k]) * alpha[k]

        return ret

    def Take_step(self, i, j, K, y, alpha, E, epsilon=1e-4):
        s = alpha[i] + alpha[j]
        L = max(-(1 - y[i]) * self.C / 2, s - (1 + y[j]) * self.C / 2)
        H = min((1 + y[i]) * self.C / 2, s + (1 - y[j]) * self.C / 2)

        if L == H:
            return False

        eta = 2 * K[i, j] - K[i, i] - K[j, j]

        if abs(eta) > 0:
            n = K.shape[0]
            alpha_new = y[j] - y[i] + s * (K[i, j] - K[j, j])
            for k in range(n):
                if k != i and k != j:
                    alpha_new += (K[i, k] - K[j, k]) * alpha[k]
            alpha_new /= eta

            if alpha_new < L:
                alpha_new = L
            elif alpha_new > H:
                alpha_new = H
        else:
            Lobj = self.Dual_objective(i, j, L, K, y, alpha)
            Hobj = self.Dual_objective(i, j, H, K, y, alpha)

            if Lobj < Hobj - epsilon:
                alpha_new = H
            elif Hobj < Lobj - epsilon:
                alpha_new = L
            else:
                alpha_new = alpha[i]

        if abs(alpha_new - alpha[i]) < 1e-5 * (alpha_new + alpha[i] + 1e-5):
            return False

        alpha[i] = alpha_new
        alpha[j] = s - alpha[i]

        E[i] = np.dot(K[i, :], alpha) + self.bias - y[i]
        E[j] = np.dot(K[j, :], alpha) + self.bias - y[j]
        b1 = -E[i] + self.bias
        b2 = -E[j] + self.bias

        if alpha[i] * y[i] > epsilon and alpha[i] * y[i] < self.C - epsilon:
            bias = b1
        elif alpha[j] * y[j] > epsilon and alpha[j] * y[j] < self.C - epsilon:
            bias = b2
        else:
            bias = (b1 + b2) / 2

        E[i] += bias - self.bias
        E[j] += bias - self.bias
        self.bias = bias
        return True

    # SMO algorithm
    def Solve_dual(self, K, y, C, iterations=50, epsilon=1e-4):
        n = y.shape[0]
        alpha = np.zeros(n)
        self.bias = 0
        self.C = C

        E = np.zeros(n)
        for i in range(n):
            E[i] = -y[i]

        loop_all = True

        for _ in range(iterations):
            num_changed = 0

            for i in range(n):
                E[i] = np.dot(K[i, :], alpha) + self.bias - y[i]
                if (loop_all \
                    or (alpha[i] * y[i] > 0 and alpha[i] * y[i] < self.C)) \
                    and ((E[i] * y[i] < -epsilon and alpha[i] * y[i] < self.C) \
                    or (E[i] * y[i] > epsilon and alpha[i] * y[i] > 0)):

                    non_bound = 0
                    for j in range(n):
                        if j != i and alpha[j] * y[j] > epsilon and alpha[j] * y[j] < self.C - epsilon:
                            non_bound = 1

                    done = False
                    if non_bound > 0:
                        k = -1
                        for j in range(n):
                            if j != i and alpha[j] * y[j] > epsilon and alpha[j] * y[j] < self.C - epsilon \
                                and (j == -1 or (E[i] > 0 and E[j] < E[k]) \
                                or (E[i] < 0 and E[j] > E[k])):
                                k = j
                        if self.Take_step(i, k, K, y, alpha, E):
                            done = True

                    if not done:
                        start = random.randint(0, n - 1)
                        for j in range(start, n):
                            if j != i and alpha[j] * y[j] > epsilon and alpha[j] * y[j] < self.C - epsilon:
                                if self.Take_step(i, j, K, y, alpha, E):
                                    done = True
                                    break

                    if not done:
                        start = random.randint(0, n - 1)
                        for j in range(start, n):
                            if j != i:
                                if self.Take_step(i, j, K, y, alpha, E):
                                    done = True
                                    break

                    if done:
                        num_changed += 1

            if loop_all:
                loop_all = False
            elif num_changed > 0:
                loop_all = True

        #print self.bias
        #result = 2 * np.dot(alpha, y) - np.dot(np.dot(alpha, K), alpha)
        #print "Final Result dual: %.8f" % result

        return alpha

    def fit(self, X, y, C, K=None, check=False):
        #print("Fit KernelSVMBinaryClassifier")
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2 and y.ndim == 1
        n = X.shape[0]

        if K is None:
            K = self.kernel.build_K(X)
        else:
            assert K.ndim == 2 and K.shape[0] == n and K.shape[1] == n

        # change y to -1/1
        self.class1 = np.min(y)
        self.class2 = np.max(y)
        assert self.class1 != self.class2
        ind1 = (y == self.class1)
        ind2 = (y == self.class2)
        y2 = np.zeros(n)
        y2[ind1] = -1
        y2[ind2] = 1

        if check:
            print("points of class %d : %d" % (self.class1, np.sum(ind1)))
            print("points of class %d : %d" % (self.class2, np.sum(ind2)))

        alpha = self.Solve_dual(K, y2, C)
        ind = (np.abs(alpha) > 1e-9)
        n_support_vectors = np.sum(ind)

        if check:
            print("support vectors: %d (of %d)" % (np.sum(ind), n))

        assert n_support_vectors > 0
        self.X = X[ind, :]
        self.alpha = alpha[ind]

        #if check:
        #    print "Accuracy on training data: %.3f" % self._calc_accuracy(X, y)

    def predict(self, X, confidence=False):
        n = X.shape[0]
        y = np.zeros(n, dtype=np.int32)

        K = self.kernel.kernel_matrix(X, self.X)
        pred = np.dot(K, self.alpha)

        for i, f in enumerate(pred):
            if confidence:
                y[i] = f + self.bias
            else:
                if f + self.bias >= 0:
                    y[i] = self.class2
                else:
                    y[i] = self.class1

        return y

    def Compute_accuracy(self, X, y,type = 'l0'):
        ypred = self.predict(X)
        return np.sum(ypred == y) * 100.0 / y.shape[0]

    def Submission(self,X_test,Submission_path):
        Predictions = self.predict(X_test)
        predicted_items_dict = {'Id':range(1,len(Predictions)+1),'Prediction':Predictions}
        dataframe_predicted_items_dict=   pd.DataFrame(predicted_items_dict)
        dataframe_predicted_items_dict.to_csv(Submission_path,index=False)
        return "Submission file construction succeeded"

class KernelSVMOneVsOneClassifier:
    """
    nclasses: number of classes (assumed between 0 and nclasses - 1)
    SVMMatrix: matrix of one vs one classifiers
    """
    def __init__(self, nclasses, kernel):
        self.nclasses = nclasses
        self.kernel = kernel
        self.SVMMatrix = []

        for i in range(self.nclasses):
            aux = []
            for _ in range(i + 1, self.nclasses):
                aux.append(KernelSVMBinaryClassifier(self.kernel))
            self.SVMMatrix.append(aux)

    def fit(self, X, y, C, validation=None, K=None, check=False):
        print("Fit KernelSVMOneVsOneClassifier")
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2 and y.ndim == 1

        n = X.shape[0]

        if validation is not None:
            assert validation > 0 and validation < 1
            split_idx = int(validation * n)
            Xval = X[:split_idx,:]
            yval = y[:split_idx]
            Xtrain = X[split_idx:,:]
            ytrain = y[split_idx:]

            if K is not None:
                K = K[split_idx:, split_idx:]
        else:
            Xtrain = X
            ytrain = y

        ind_by_class = []
        for i in range(self.nclasses):
            ind = (ytrain == i)
            ind_by_class.append(ind)

        if K is None:
            K = self.kernel.build_K(Xtrain)

        pbar = tqdm(total=self.nclasses * (self.nclasses - 1) / 2)

        for i in range(self.nclasses):
            for j in range(i + 1, self.nclasses):
                ind = np.logical_or(ind_by_class[i], ind_by_class[j])
                partial_K = K[ind, :]
                partial_K = partial_K[:, ind]

                self.SVMMatrix[i][j - i - 1].fit(Xtrain[ind, :], ytrain[ind], C, K=partial_K, check=check)
                pbar.update(1)
        pbar.close()

        if validation is not None:
            accuracy = self.Compute_accuracy(Xval, yval)
            print("Accuracy in validation data is %.3f" % accuracy)

    def predict(self, X):
        n = X.shape[0]
        scores = np.zeros((n, self.nclasses))

        print("One vs One prediction")
        pbar = tqdm(total=self.nclasses * (self.nclasses - 1) / 2)
        for i in range(self.nclasses):
            for j in range(i + 1, self.nclasses):
                y = self.SVMMatrix[i][j - i - 1].predict(X)

                for k, pred in enumerate(y):
                    scores[k][pred] += 1

                pbar.update(1)
        pbar.close()

        return np.argmax(scores, axis=1)

    def Compute_accuracy(self, X, y,type = 'l0'):
        test_list = (self.predict(X) == y)
        return len(test_list[test_list == True]) / len(y)

    def Submission(self,X_test,Submission_path):
        Predictions = self.predict(X_test)
        predicted_items_dict = {'Id':range(1,len(Predictions)+1),'Prediction':Predictions}
        dataframe_predicted_items_dict=   pd.DataFrame(predicted_items_dict)
        dataframe_predicted_items_dict.to_csv(Submission_path,index=False)
        return "Submission file construction succeeded"


class KernelSVMOneVsAllClassifier:
    """
    nclasses: number of classes (assumed between 0 and nclasses - 1)
    SVMova: list of one vs all classifiers
    """
    def __init__(self, nclasses, kernel):
        self.nclasses = nclasses
        self.kernel = kernel
        self.SVMova = []

        for _ in range(self.nclasses):
            self.SVMova.append(KernelSVMBinaryClassifier(self.kernel))

    def fit(self, X, y, C, validation=None, K=None, check=False):
        print("Fit KernelSVMOneVsAllClassifier")
        assert X.shape[0] == y.shape[0]
        assert X.ndim == 2 and y.ndim == 1

        n = X.shape[0]

        if validation is not None:
            assert validation > 0 and validation < 1
            split_idx = int(validation * n)
            Xval = X[:split_idx,:]
            yval = y[:split_idx]
            Xtrain = X[split_idx:,:]
            ytrain = y[split_idx:]

            if K is not None:
                K = K[split_idx:, split_idx:]
        else:
            Xtrain = X
            ytrain = y

        if K is None:
            K = self.kernel.build_K(Xtrain)

        for i in tqdm(range(self.nclasses)):
            y2 = -np.ones(ytrain.shape[0])
            ind = (ytrain == i)
            y2[ind] = 1
            self.SVMova[i].fit(Xtrain, y2, C, K=K, check=check)

        if validation is not None:
            accuracy = self.Compute_accuracy(Xval, yval)
            print("Accuracy in validation data is %.3f" % accuracy)

    def predict(self, X):
        n = X.shape[0]
        scores = np.zeros((n, self.nclasses))
        print("One vs All prediction")
        for i in tqdm(range(self.nclasses)):
            scores[:, i] = self.SVMova[i].predict(X, confidence=True)
        return np.argmax(scores, axis=1)

    def Compute_accuracy(self, X, y,type = 'l0'):
        test_list = (self.predict(X) == y)
        return len(test_list[test_list == True]) / len(y)

    def Submission(self, X_test, Submission_path):
        Predictions = self.predict(X_test)
        predicted_items_dict = {'Id': range(1, len(Predictions) + 1), 'Prediction': Predictions}
        dataframe_predicted_items_dict = pd.DataFrame(predicted_items_dict)
        dataframe_predicted_items_dict.to_csv(Submission_path, index=False)
        return "Submission file construction succeeded"