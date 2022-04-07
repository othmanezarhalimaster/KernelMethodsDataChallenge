# Kernel Methods Data challenge

import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd
import sys
import time

import warnings
warnings.filterwarnings("ignore")

# Run visualization ====================================================================================================
class Helper:

    @staticmethod
    def log_process(title, cursor, finish_cursor, start_time=None):
        percentage = float(cursor + 1) / finish_cursor
        now_time = time.time()
        time_to_finish = ((now_time - start_time) / percentage) - (now_time - start_time)
        mn, sc = int(time_to_finish // 60), int((time_to_finish / 60 - time_to_finish // 60) * 60)
        if start_time:
            sys.stdout.write(
                "\r%s - %.2f%% ----- remaining time: %d min %d sec -----" % (title, 100 * percentage, mn, sc))
            sys.stdout.flush()
        else:
            sys.stdout.write("\r%s - \r%.2f%%" % (title, 100 * percentage))
            sys.stdout.flush()


# Data acquisition =====================================================================================================
class Dataset:
    def __init__(self,path_XTrain,path_YTrain,path_XTest):
        self.path_XTrain = path_XTrain
        self.path_YTrain = path_YTrain
        self.path_XTest = path_XTest
        self.Xtrain_dataframe = None
        self.Xtest_dataframe = None
        self.Ytrain_dataframe = None
        self.Xtrain = None
        self.Xtest = None
        self.Ytrain = None

    def DatasetConstruction(self,datasize=-1, length=-1):
        Xtrain, Ytrain, Xtest = pd.read_csv(self.path_XTrain, sep=',',header=None), pd.read_csv(self.path_YTrain, sep=','), pd.read_csv(
            self.path_XTest, sep=',',header=None)
        Xtrain, Xtest = Xtrain.drop(Xtrain.columns[[-1]], axis=1), Xtest.drop(Xtest.columns[[-1]], axis=1)
        self.Xtrain_dataframe, self.Xtest_dataframe, self.Ytrain_dataframe = Xtrain, Xtest, Ytrain
        if length != -1:
            Xtrain, Xtest = Xtrain.drop(Xtrain.columns[length:], axis=1), Xtest.drop(Xtest.columns[length:], axis=1)
        Ytrain = Ytrain.drop('Id', axis=1)['Prediction'].to_numpy()
        if datasize != -1:
            Xtrain, Ytrain = Xtrain[:datasize], Ytrain[:datasize]
        Xtrain, Xtest = Xtrain.to_numpy(), Xtest.to_numpy()
        self.Xtrain,self.Xtest,self.Ytrain = Xtrain,Xtest,Ytrain
        return {'train': {'x': self.Xtrain, 'y': self.Ytrain}, 'test': {'x': self.Xtest}}

    def Scale(self):
        mean_train,stdev_train = self.Xtrain.mean(),self.Xtrain.std()
        self.Xtrain = (self.Xtrain-mean_train*np.ones((len(self.Xtrain),len(self.Xtrain[0]))))/stdev_train
        return {'train': {'x': self.Xtrain, 'y': self.Ytrain}, 'test': {'x': self.Xtest}}

    def Image_visualizer(self,pixelsraw, size, colordepth=3):
        image = np.resize(1 * pixelsraw, (size, size, colordepth))
        plt.imshow(image.astype('uint8'))
        plt.show()
        return 'Image visualization succeeded'

    def color_train_preprocessing(self,size=-1):
        Xtrain,Xtest = self.Xtrain_dataframe.values,self.Xtest_dataframe.values
        ytrain = self.Ytrain_dataframe['Prediction']
        if size!=-1:
            Xtrain,ytrain = Xtrain[:size],ytrain[:size]
        red_train, green_train, blue_train = np.hsplit(Xtrain, 3)
        red_test, green_test, blue_test = np.hsplit(Xtest, 3)
        data_train = np.array([np.dstack((red_train[i], red_train[i], red_train[i])).reshape(32, 32, 3) for i in range(len(Xtrain))])
        data_test = np.array([np.dstack((red_test[i], blue_test[i], green_test[i])).reshape(32, 32, 3) for i in range(len(Xtest))])
        return data_train,data_test,ytrain

    def data_augmention(self,size=-1):
        start = time.time()
        color_processed_data = self.color_train_preprocessing(size)
        data_train,ytrain =  color_processed_data[0],color_processed_data[2]
        length = len(data_train)
        augmented_train = []
        for row in range(0, length):
            if row % 50 == 0 or row == length - 1:
                Helper.log_process('Flipping image...', row, finish_cursor=length, start_time=start)
            augmented_train.append(data_train[row])
            augmented_train.append(ImageTransformation.flip_image_horizontal(data_train[row]))
        augmented_train = np.array(augmented_train)
        # Compute augmented labels
        augmented_labels = []
        for row in range(length):
            aux = ytrain[row]
            augmented_labels.append(aux)
            augmented_labels.append(aux)
        augmented_labels = np.array(augmented_labels)
        return augmented_train,augmented_labels




class ImageTransformation:
    def __init__(self,designation='horizontal transform'):
        self.designation = designation

    def flip_image_horizontal(image):
        # Takes an image as input and outputs the same image with a horizontal flip
        result = image.copy()
        for channel in range(3):
            aux = image[:, :, channel]
            for column in range(len(aux)):
                result[:, column, channel] = aux[:, len(aux) - column - 1]
        return result