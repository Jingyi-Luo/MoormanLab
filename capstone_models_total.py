#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rc('font', size=14)
import numpy as np
#from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

#
working_dir = '/Users/ljyi/Desktop/capstone/A_capstone_data'
os.chdir(working_dir)
#
raw_data = pd.read_csv('moss_plos_one_data.csv')
raw_data.columns = raw_data.columns.str.replace('.', '_')
#raw_data.columns.tolist()
#raw_data.head()
#type(raw_data)
raw_data.shape
# (2217958, 62)
col_names = raw_data.columns.tolist()
# list(raw_data.columns)
#raw_data['id'].unique()

#==============================================================================
#                             Data Preprocessing
#==============================================================================
# find missing values
df = raw_data
df.head()
df_nan = df.isnull().sum(axis=0).to_frame()
df_nan.columns=['counts']
col_nan = df_nan[df_nan['counts']>0]
col_nan_index = list(col_nan.index)

# find unique values in 'id'
id_unique = df['id'].unique().tolist()
id_unique
len(id_unique)
# 8105

# get train and test index based on unique 'id'
import random
random.seed(1)
train_id = random.sample(id_unique, 5674)
len(train_id)
type(train_id)
# 5674
test_id = [avar for avar in id_unique if avar not in train_id]
len(test_id)
# 2431

# get rid of variables with two many missing values
data_df = raw_data
drop_cols = ['n_evts', 'LOS', 'ICU_Pt_Days', 'Mort', 'age']  # can we keep 'age'?
data_df.drop(col_nan_index, inplace=True, axis=1)
data_df.drop(drop_cols, inplace=True, axis=1)
data_df.shape
# (2217958, 52)

# 'race' with three levels and 'svc' with four levels are categorical data
dummy_race = pd.get_dummies(data_df['race'])
data_df_dummy = pd.concat([data_df, dummy_race], axis=1)
data_df_dummy.shape
#type(data_df_dummy)
# (2217958, 55)
data_df_dummy.drop(columns=['race', 'oth'], inplace=True, axis=1) # dummy variable trap
data_df_dummy.shape
# (2217958, 53)

dummy_svc = pd.get_dummies(data_df['svc'])
df_svc_dummy = pd.concat([data_df_dummy, dummy_svc], axis=1)
df_svc_dummy.shape
# (2217958, 57)
df_svc_dummy.drop(columns=['svc', 'Other'], inplace=True, axis=1)
df_svc_dummy.shape
# (2217958, 55)
list(df_svc_dummy.columns)
df_dummy = df_svc_dummy

# split data into training and testing sets
df_dummy.set_index('id', inplace=True)
X_y_train = df_dummy.loc[train_id]
X_y_test = df_dummy.loc[test_id]
# 
true_index = np.where(X_y_train['y'].values.flatten() == True)[0]
false_index = np.where(X_y_train['y'].values.flatten() == False)[0]
random.seed(0)
selected_false_index = random.sample(list(false_index), len(true_index)*2)
train_index = list(np.append(true_index, selected_false_index))
#
true_index = np.where(X_y_test['y'].values.flatten() == True)[0]
false_index = np.where(X_y_test['y'].values.flatten() == False)[0]
random.seed(0)
selected_false_index = random.sample(list(false_index), len(true_index)*2)
test_index = list(np.append(true_index, selected_false_index))
#
X_train = X_y_train.iloc[train_index, X_y_train.columns != 'y']
#X_train.drop('id', axis=1, inplace=True)
y_train = X_y_train.iloc[train_index, X_y_train.columns == 'y']
X_test = X_y_test.iloc[:, X_y_test.columns != 'y']
#X_test.drop('id', axis=1, inplace=True)
y_test = X_y_test.iloc[:, X_y_test.columns == 'y']
y_test = y_test.values.flatten()
#X_train = X_y_train.drop('y', 1)
#y_train = X_y_train['y']  
#print(i)
len(y_train)
#1520840
np.sum(y_train == True)
# 16391
np.sum(y_train == False)
# 1504449
np.sum(y_test == True)
# 7490
np.sum(y_test == False)
# 689628

train_col_names = X_train.columns

# over-sampling using SMOTE-Synthetic Minority Oversampling Technique
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
os_data_X, os_data_y = os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X, columns=train_col_names)
os_data_y = pd.DataFrame(data=os_data_y, columns=['y'])
# check the lengths of data now
os_data_X.shape
# (2996702, 55)
len(os_data_y)
# 2996702
# percent of True
n_total = len(os_data_y)
n_true = sum(os_data_y['y']==True)
n_true
# 1498351 (before oversampling: 23881)
# increased number of true events
1535746-23881
# 1511865
n_false = sum(os_data_y['y']==False)
n_false
# 1498351 (before oversampling:2194077)
# decreased number of false events 
1535746-2194077
# -658331
pct_true = n_true/n_total
pct_true
# 0.5
# 50% are event
pct_false = n_false/n_total
pct_false
# 0.5
# 50% are non-event
# here, the ratio of event to non-event is 1:1 after SMOTE.

# Final data for training 
X_train_balanced = os_data_X
y_train_balanced = os_data_y
len(X_train_balanced)
# 2996702
len(y_train_balanced)
# 2996702
n_rows_total = len(y_train_balanced)
n_rows_total_ls = range(n_rows_total)
random.seed(1)
#sample_rows_index = random.sample(n_rows_total_ls, 100000)
X_train_df = X_train_balanced
y_train_sample = y_train_balanced
y_train_sample = y_train_sample.values.flatten()
len(X_train_df)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sample = sc.fit_transform(X_train_df)  
X_test = sc.transform(X_test)

#==============================================================================
#                             Logistic Regression
#==============================================================================
# Instantiate the model using default parameters
logreg = LogisticRegression(random_state=0)
# fit the model with X_train, y_train
logreg.fit(X_train_sample, y_train_sample)

# prediction X_train
y_train_pred_logreg = logreg.predict(X_train_sample)
# evaluate model: accuracy of logitstic regression classifier on training set
#logreg.score(X_train_sample, y_train_sample)
# 0.69025
#metrics.accuracy_score(y_train_sample, y_train_pred_logreg)

# evaluate model: Confusion Matrix
from sklearn import metrics
cnf_matrix_train = metrics.confusion_matrix(y_train_sample, y_train_pred_logreg)
cnf_matrix_train
# array([[36195, 13837],
#       [17138, 32830]])
# diagonal values represent accurate predictions, 
# non-diagonal elements are inaccurate predictions.
# the accuracy rate:
accurate_rate_logreg_train = (cnf_matrix_train[0,0]+cnf_matrix_train[1,1])/len(y_train_sample)
accurate_rate_logreg_train
# 0.68283

# prediction X_test 
y_pred_logreg = logreg.predict(X_test)
# evaluate model: accuracy of logitstic regression classifier on test set
#metrics.accuracy_score(y_test, y_pred_logreg)
# 0.7085858876824228

# Score 1
# Confusion Matrix
cnf_matrix_logreg = metrics.confusion_matrix(y_test, y_pred_logreg)
#accurate_rate_logreg = (cnf_matrix[0,0]+cnf_matrix[1,1])/len(y_test)
#accurate_rate_logreg
# 0.7085858876824228
Se_logreg = cnf_matrix_logreg[0, 0]/(cnf_matrix_logreg[0, 0]+cnf_matrix_logreg[1, 0])
p_plus_logreg = cnf_matrix_logreg[0, 0]/(cnf_matrix_logreg[0, 0]+cnf_matrix_logreg[0, 1])
score_1_logreg = min(Se_logreg, p_plus_logreg)
score_1_logreg
# 0.709561751154195

# ROC Curve: the receiver operating characteristic curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
y_pred_proba_logreg = logreg.predict_proba(X_test)[:, 1]
logit_roc_auc = roc_auc_score(y_test, y_pred_proba_logreg)
logit_roc_auc
# 0.7164077427323496
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_logreg)
# logreg.classes_
# array([False,  True])
plt.figure()
plt.plot(fpr, tpr, label = "Logistic Regression (area = {:0.2f})".format(logit_roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operatting Characteristic: Logistic Regression')
plt.legend(loc='lower right')
plt.savefig('Log_ROC')
plt.show()

# summary:
# Score 1: 0.710
# AUC: 0.716

#==============================================================================
#                             Random Forest
#==============================================================================
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# create a random forest classifer
forest_clf = RandomForestClassifier(n_estimators=600, n_jobs = -1, random_state=0)
# train the classifer using the training set
forest_clf.fit(X_train_sample, y_train_sample)

# prediction X_train
y_train_pred_forest = forest_clf.predict(X_train_sample)
metrics.accuracy_score(y_train_sample, y_train_pred_forest)

# prediction X_test 
y_pred_forest = forest_clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred_forest)

# Score 1
cnf_matrix_forest = metrics.confusion_matrix(y_test, y_pred_forest)
Se_forest = cnf_matrix_forest[0, 0]/(cnf_matrix_forest[0, 0]+cnf_matrix_forest[1, 0])
p_plus_forest = cnf_matrix_forest[0, 0]/(cnf_matrix_forest[0, 0]+cnf_matrix_forest[0, 1])
score_1_forest = min(Se_forest, p_plus_forest)
score_1_forest
# 0.709561751154195

# ROC Curve: the receiver operating characteristic curve
y_pred_proba_forest = forest_clf.predict_proba(X_test)[:, 1]
forest_roc_auc = roc_auc_score(y_test, y_pred_proba_forest)
forest_roc_auc
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_forest)

plt.figure()
plt.plot(fpr, tpr, label = "Random Forest Regression (area = {:0.3f})".format(forest_roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operatting Characteristic (600 trees)')
plt.legend(loc='lower right')
plt.savefig('Forest_ROC_600_trees')
plt.show()

# Summary:
# 100000 obs of X_train_sample
# trees      AUC                   Score-1
# 50         0.666                  
# 80         0.682                  
# 100        0.685                  
# 200        0.699                    
# 300        0.703                         
# 400        0.704                      
# 500        0.707                          
# 600        0.707                      

# feature importance with random forest
imp_feat_rf = pd.Series(forest_clf.feature_importances_, index=X_train_sample.columns.tolist()).sort_values(ascending=False) #????????????????????????????????????????????????
imp_feat_rf.plot(kind='bar', title='Feature Importance with Random Forest', color='b', figsize=(12, 8))
plt.ylabel('Feature Importance Values')
plt.subplots_adjust(bottom=0.25)
plt.savefig('FeatureImportance.png')
plt.show()

imp_feat_rf.to_csv('Feature Importance.csv')
 
#==============================================================================
#                        Support Vector Machine
#==============================================================================

# SVM does not provide probability estimates directly.
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#from sklearn.metrics import classification_report

# Linear svm classifier -------------------------------------------------------
svm_linear_clf = SVC(kernel='linear', probability=True, random_state=0)
# train model
svm_linear_clf.fit(X_train_sample, y_train_sample)

# prediction X_train
y_train_pred_svm_linear = svm_linear_clf.predict(X_train_sample)
metrics.accuracy_score(y_train_sample, y_train_pred_svm_linear)

# prediction X_test & accuracy score
y_pred_svm_linear = svm_linear_clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred_svm_linear)

## using classification_report to get precision, recall and F1 
#classification_report(y_test, y_pred_svm)

# Score-1
cnf_matrix_svm_linear = metrics.confusion_matrix(y_test, y_pred_svm_linear)
Se_svm_linear = cnf_matrix_svm_linear[0, 0]/(cnf_matrix_svm_linear[0, 0]+cnf_matrix_svm_linear[1, 0])
p_plus_svm_linear = cnf_matrix_svm_linear[0, 0]/(cnf_matrix_svm_linear[0, 0]+cnf_matrix_svm_linear[0, 1])
score_1_svm_linear = min(Se_svm_linear, p_plus_svm_linear)
score_1_svm_linear
# 

# ROC Curve
y_pred_proba_svm_linear = svm_linear_clf.predict_proba(X_test)[:, 1]
svm_linear_roc_auc = roc_auc_score(y_test, y_pred_proba_svm_linear)
svm_linear_roc_auc
#
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_svm_linear)

plt.figure()
plt.plot(fpr, tpr, label = "Support Vector Machine (area = {:0.3f})".format(svm_linear_roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operatting Characteristic: SVM_linear')
plt.legend(loc='lower right')
plt.savefig('SVM_ROC_linear')
plt.show()

# Non-linear svm classifier ---------------------------------------------------
svm_kernel_clf = SVC(kernel='rbf', probability=True, random_state=0)
svm_kernel_clf.fit(X_train_sample, y_train_sample)

# prediction X_train
y_train_pred_svm_kernel = svm_kernel_clf.predict(X_train_sample)
metrics.accuracy_score(y_train_sample, y_train_pred_svm_kernel)

# prediction X_test & accuracy score
y_pred_svm_kernel = svm_kernel_clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred_svm_kernel)
# 

# Confusion Matrix
#cnf_matrix_svm_kernel = metrics.confusion_matrix(y_test, y_pred_svm_kernel)
# the accuracy rate:
#accurate_rate_svm_kernel = (cnf_matrix_svm_kernel[0,0]+cnf_matrix_svm_kernel[1,1])/len(y_test)

# Score-1
cnf_matrix_svm_kernel = metrics.confusion_matrix(y_test, y_pred_svm_kernel)
Se_svm_kernel = cnf_matrix_svm_kernel[0, 0]/(cnf_matrix_svm_kernel[0, 0]+cnf_matrix_svm_kernel[1, 0])
p_plus_svm_kernel = cnf_matrix_svm_kernel[0, 0]/(cnf_matrix_svm_kernel[0, 0]+cnf_matrix_svm_kernel[0, 1])
score_1_svm_kernel = min(Se_svm_kernel, p_plus_svm_kernel)
score_1_svm_kernel
# 

# ROC Curve
y_pred_proba_svm_kernel = svm_kernel_clf.predict_proba(X_test)[:, 1]
svm_kernel_roc_auc = roc_auc_score(y_test, y_pred_proba_svm_kernel)
svm_kernel_roc_auc
#
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_svm_kernel)

plt.figure()
plt.plot(fpr, tpr, label = "Kernel Support Vector Machine (area = {:0.3f})".format(svm_kernel_roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operatting Characteristic: Kernel SVM')
plt.legend(loc='lower right')
plt.savefig('Kernel_SVM_ROC')
plt.show()

# Summary:
# Kernel          AUC      Score-1
# Linear
# Gaussian

#==============================================================================
#                                   KNN
#==============================================================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

knn_clf = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2) #Euclidean
knn_clf.fit(X_train_sample, y_train_sample)

# prediction X_train
y_train_pred_knn = knn_clf.predict(X_train_sample)
metrics.accuracy_score(y_train_sample, y_train_pred_knn)
# 

# prediction X_test & accuracy score
y_pred_knn = knn_clf.predict(X_test)
metrics.accuracy_score(y_test, y_pred_knn)

# Score 1
cnf_matrix_knn = metrics.confusion_matrix(y_test, y_pred_knn)
#accurate_rate_knn = (cnf_matrix_knn[0,0]+cnf_matrix_knn[1,1])/len(y_test)
Se_knn = cnf_matrix_knn[0, 0]/(cnf_matrix_knn[0, 0]+cnf_matrix_knn[1, 0])
p_plus_knn = cnf_matrix_knn[0, 0]/(cnf_matrix_knn[0, 0]+cnf_matrix_knn[0, 1])
score_1_knn = min(Se_knn, p_plus_knn)
score_1_knn
# 

# ROC Curve
y_pred_proba_knn = knn_clf.predict_proba(X_test)[:, 1]
knn_roc_auc = roc_auc_score(y_test, y_pred_proba_knn)
knn_roc_auc
#
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_knn)

plt.figure()
plt.plot(fpr, tpr, label = "KNN (area = {:0.2f})".format(knn_roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operatting Characteristic: KNN')
plt.legend(loc='lower right')
plt.savefig('KNN_ROC')
plt.show()

# Summary:
# n_neighbor      AUC        Score-1
#   5

#==============================================================================
#                          Neural Network
#==============================================================================
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sample = sc.fit_transform(X_train_sample)
X_test = sc.transform(X_test)

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential # initialize ANN
from keras.layers import Dense # build the layers of ANN (input, hidden, output...)

X_train_sample.shape
# 10000, 53

# Initialising the ANN
ANN_clf = Sequential()
# Adding the input layer and the first hidden layer
ANN_clf.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu', input_dim = len(X_train_sample[0]))) # six neuron， 55 features
# Adding the second hidden layer
ANN_clf.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu')) # uniformly set weight, 'relu': activation function is rectifier
# Adding the output layer
ANN_clf.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid')) # True or False, binary, so output_dim is 1. Output function is sigmoid.
# Compiling the ANN
ANN_clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # 'adam'optimize weight. Loss: cost. Use accuracy as metrics.
# Fitting the ANN to the Training set
ANN_clf.fit(X_train, y_train, batch_size = 10, nb_epoch = 100) # after input 10 obs, then adjust weight. Epoch: circulate 2 times to adjust weight and accuracy.

# Predicting the Test set results
y_pred_proba_ANN = ANN_clf.predict(X_test)   

# Score 1
y_pred_ANN = (y_pred_proba_ANN > 0.5)
#accuracy_ANN = (cm[0,0] +cm[1,1])/len(y_test)
cnf_matrix_ANN = metrics.confusion_matrix(y_test, y_pred_ANN)
Se_ANN = cnf_matrix_ANN[0, 0]/(cnf_matrix_ANN[0, 0]+cnf_matrix_ANN[1, 0])
p_plus_ANN = cnf_matrix_ANN[0, 0]/(cnf_matrix_ANN[0, 0]+cnf_matrix_ANN[0, 1])
score_1_ANN = min(Se_ANN, p_plus_ANN)
score_1_ANN
# 

# ROC Curve: the receiver operating characteristic curve
y_pred_proba_ANN = ANN_clf.predict(X_test) 
ANN_roc_auc = roc_auc_score(y_test, y_pred_proba_ANN)
ANN_roc_auc
# 0.6938920358342346
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_ANN)
# logreg.classes_
# array([False,  True])
plt.figure()
plt.plot(fpr, tpr, label = "Artificial Neural Network (area = {:0.2f})".format(ANN_roc_auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operatting Characteristic: Artificial Neural Network')
plt.legend(loc='lower right')
plt.savefig('ANN_ROC')
plt.show()

# Summary:
# Hidden layers    units   batch-size     epoch       AUC       Score-1
#  2                64       10            100

#==============================================================================
