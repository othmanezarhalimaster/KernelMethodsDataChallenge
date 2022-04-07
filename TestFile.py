# Kernel Methods Data challenge


# Package importation
from BrutForceKSVM import *
from DataProcessing import *
from Kernels import *



# Data acquisition =====================================================================================================
path_XTrain = "C://Users//data//Desktop//Kproject//Xtr.csv"
path_YTrain = "C://Users//data//Desktop//Kproject//Ytr.csv"
path_XTest = "C://Users//data//Desktop//Kproject//Xte.csv"
Submission_path = "C://Users//data//Desktop//Kproject//Submission.csv"

Datasetclass = Dataset(path_XTrain,path_YTrain,path_XTest)
Final_Instance_Dataset = Datasetclass.DatasetConstruction(20)
#Final_Instance_Dataset = Datasetclass.Scale()

numclasses = 10

# Model trainer ========================================================================================================

#C = 1000.
C=5.5
# Linear kernel
# kernel = Linear()
# model = KernelMSVC(C=C, kernel=kernel)
# train_dataset = Final_Instance_Dataset['train']
# model.fit(train_dataset['x'], train_dataset['y'])
# print(model.l0Accuracy(Final_Instance_Dataset['train']['x'],Final_Instance_Dataset['train']['y']))

# Gaussian kernel
# kernel = RBF(sqrt(1e+3))
# model = KernelMSVC(C=C, kernel=kernel)
# train_dataset = Final_Instance_Dataset['train']
# model.fit(train_dataset['x'], train_dataset['y'])
# print(model.l0Accuracy(Final_Instance_Dataset['train']['x'],Final_Instance_Dataset['train']['y']))
# print(model.Submission(Final_Instance_Dataset['test']['x']))

# tanh  kernel
# kernel = Sigmoid(0.5,-2)
# model = KernelMSVC(C=C, kernel=kernel)
# train_dataset = Final_Instance_Dataset['train']
# model.fit(train_dataset['x'], train_dataset['y'])
# print(model.l0Accuracy(Final_Instance_Dataset['train']['x'],Final_Instance_Dataset['train']['y']))
# print(model.Submission(Final_Instance_Dataset['test']['x']))

# Polynomial  kernel
# kernel = PolynomialKernel(3,-2)
# model = KernelMSVC(C=C, kernel=kernel)
# train_dataset = Final_Instance_Dataset['train']
# model.fit(train_dataset['x'], train_dataset['y'])
# print(model.l0Accuracy(Final_Instance_Dataset['train']['x'],Final_Instance_Dataset['train']['y']))
# print(model.Submission(Final_Instance_Dataset['test']['x']))

# GHI  kernel
# kernel = GHI_Kernel(3)
# model = KernelMSVC(C=C, kernel=kernel)
# train_dataset = Final_Instance_Dataset['train']
# model.fit(train_dataset['x'], train_dataset['y'])
# print(model.l0Accuracy(Final_Instance_Dataset['train']['x'],Final_Instance_Dataset['train']['y']))
# print(model.Submission(Final_Instance_Dataset['test']['x']))


########################################################################################################################
# MKL Gaussian
# kernellist,kernelweights = [RBF(5),RBF(2),RBF(1),RBF(1.5)] ,[0.2,0.2,0.40,0.20]
# kernel = MKL(kernellist,kernelweights)
# model = KernelMSVC(C=C, kernel=kernel)
# train_dataset = Final_Instance_Dataset['train']
# model.fit(train_dataset['x'], train_dataset['y'])
# print(model.l0Accuracy(Final_Instance_Dataset['train']['x'],Final_Instance_Dataset['train']['y']))
# print(model.Submission(Final_Instance_Dataset['test']['x']))

#MKL Linear
# kernellist,kernelweights = [Linear(),Linear(),Linear(),Linear() ] ,[0.2,0.2,0.40,0.2]
# kernel = MKL(kernellist,kernelweights)
# model = KernelMSVC(C=C, kernel=kernel)
# train_dataset = Final_Instance_Dataset['train']
# model.fit(train_dataset['x'], train_dataset['y'])
# print(model.l0Accuracy(Final_Instance_Dataset['train']['x'],Final_Instance_Dataset['train']['y']))
# print(model.Submission(Final_Instance_Dataset['test']['x']))

#MKL RBF + poly
# kernellist,kernelweights = [RBF(sqrt(1e+3)),PolynomialKernel(3,-2)] ,[0.5,0.5]
# kernel = MKL(kernellist,kernelweights)
# model = KernelMSVC(C=C, kernel=kernel)
# train_dataset = Final_Instance_Dataset['train']
# model.fit(train_dataset['x'], train_dataset['y'])
# print(model.l0Accuracy(Final_Instance_Dataset['train']['x'],Final_Instance_Dataset['train']['y']))
# print(model.Submission(Final_Instance_Dataset['test']['x']))


# Results   FALSE RESULTS
#
# - RBF accuracy 0.905/datasize 200
# - Linear accuracy 0.89/datasize 200
# - MKL [RBF(5),RBF(2),RBF(1),RBF(1.5)] [0.2,0.2,0.40,0.20] accuracy 0.955/datasize 200  0.913/datasize 300  1.0/datasize 400

# MKL linear: 150 accuracy 0.59

########################################################################################################################
# FEATURE EXTRACTION - SMO approach

from KSVM_SMO import *
from FeatureExtraction import *


extractor = SIFT(gs=6,
                 ps=31,
                 sift_thres=.3,
                 gaussian_sigma=.4,
                 gaussian_thres=.7,
                 num_angles=12,
                 num_bins=5,
                 alpha=9)
Xtrain,Ytrain,Xtest = Final_Instance_Dataset['train']['x'],Final_Instance_Dataset['train']['y'],Final_Instance_Dataset['test']['x']


#with_feature_extraction = True
with_augmented_data = True

###### FEATURE EXTRACTION ##############################################################################################

# if with_feature_extraction:
#     Color_processed_data =Datasetclass.color_train_preprocessing()
#     TrainData = extractor.get_X(Color_processed_data[0])
#     TestData = extractor.get_X(Color_processed_data[1])
#     ytrain = Color_processed_data[2]

if with_augmented_data:
    augmented_train,augmented_labels = Datasetclass.data_augmention()
    TrainData = extractor.get_X(augmented_train)
    TestData = extractor.get_X(Datasetclass.color_train_preprocessing()[1])
    ytrain = augmented_labels
###### K CLASSIFICATION ################################################################################################

Classifier = 'svm_ovo'
#validation_limit = 2000
validation = 0.2
Kernel_type = 'chi_MKL'
# chi ovo acc 0.9992
# chi ova acc 0.6692
# MKL chi ovo 0.9992 / leaderboard score: 0.603   [Chi_Kernel(0.6), Chi_Kernel(0.7), Chi_Kernel(0.4), Chi_Kernel(0.2)], [0.25, 0.25,0.25, 0.25]
do_validation,do_prediction = False,True

do_submission = True

svm_kernel = Linear()
if Kernel_type == 'RBF':
    svm_kernel = RBF()
elif Kernel_type == 'chi':
    svm_kernel = Chi_Kernel(0.6)
elif Kernel_type == 'GHI':
    svm_kernel = GHI_Kernel(3)
elif Kernel_type == 'chi_MKL':
    kernellist, kernelweights = [Chi_Kernel(0.6), Chi_Kernel(0.7), Chi_Kernel(0.4), Chi_Kernel(0.2)], [0.25, 0.25,0.25, 0.25]
    svm_kernel = MKL(kernellist,kernelweights)


if Classifier == 'svm_ovo':
    K = svm_kernel.kernel_matrix(TrainData, TrainData)
    model = KernelSVMOneVsOneClassifier(numclasses, svm_kernel)

    if do_validation:
        model.fit(TrainData, ytrain, C, validation, K=K, check=True)

    if do_prediction:
        model.fit(TrainData, ytrain, C, K=K)
        print('model accuracy')
        print(model.Compute_accuracy(TrainData, ytrain))

    if do_submission:
        model.Submission(TestData, Submission_path)

elif Classifier == 'svm_ova':
    K = svm_kernel.kernel_matrix(TrainData, TrainData)
    model = KernelSVMOneVsAllClassifier(numclasses, svm_kernel)

    if do_validation:
        model.fit(TrainData, ytrain, C, validation, K=K, check=True)

    if do_prediction:
        model.fit(TrainData, ytrain, C, K=K)
        print('model accuracy')
        print(model.Compute_accuracy(TrainData, ytrain))

    if do_submission:
        model.Submission(TestData, Submission_path)





