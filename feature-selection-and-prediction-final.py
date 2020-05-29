import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from collections import OrderedDict
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import os


#%% Improting pre-process train and test datasets
df_train = pd.read_csv('trainCleaned.csv')
df_test = pd.read_csv('testCleaned.csv')

#%%creating the response and features for train and test datasets 
response_train=df_train['satisfied']
features_train=df_train.drop('satisfied',1)
features_names=features_train.columns
features_test=df_test.copy()


#%% scaling features using min-max scaler
from sklearn.preprocessing import MinMaxScaler
minmax_scaler=MinMaxScaler()

features_train_scaled=minmax_scaler.fit_transform(features_train.values.astype(float))
features_train_scaled=pd.DataFrame(features_train_scaled,columns=features_names)

features_test_scaled=minmax_scaler.fit_transform(features_test.values.astype(float))
features_test_scaled=pd.DataFrame(features_test_scaled,columns=features_names)

#%% Correltion heating map for some important features
feature_corr=[97, 34, 35, 71, 73, 51, 83, 85, 86, 87, 88, 121, 123]
df_corr=df_train.iloc[:,feature_corr]
df_corr['satisfied']=df_train['satisfied']
corr=df_corr.corr()
#%%
plt.figure(1,figsize=(10,6))
sns.heatmap(corr, xticklabels=df_corr.columns, yticklabels=df_corr.columns, cmap='Blues')
plt.xlabel('Features')
#plt.ylabel('Features')
plt.savefig('heatmap.svg')
plt.show

#%%
size0=20
size1=15
plt.figure(3,figsize=(7.5,10))
#--------------------
plt.subplot(231)
plt.hist(features_train_scaled['v237'], color='olive')
plt.xticks(fontsize=size1)
plt.yticks(fontsize=size1)
plt.title('Histogram of v237',fontsize=size0)
#--------------------
plt.subplot(232)
plt.hist(features_train_scaled['v223'], color='olive')
plt.xticks(fontsize=size1)
plt.yticks(fontsize=size1)
plt.title('Histogram of v223',fontsize=size0)
#--------------------
plt.subplot(233)
plt.hist(features_train_scaled['v180'], color='olive')
plt.xticks(fontsize=size1)
plt.yticks(fontsize=size1)
plt.title('Histogram of v180',fontsize=size0)
#--------------------
plt.subplot(234)
plt.hist(features_train_scaled['v178'], color='olive')
plt.xticks(fontsize=size1)
plt.yticks(fontsize=size1)
plt.title('Histogram of v178',fontsize=size0)
#--------------------
plt.subplot(235)
plt.hist(features_train_scaled['v225'], color='olive')
plt.xticks(fontsize=size1)
plt.yticks(fontsize=size1)
plt.title('Histogram of v225',fontsize=size0)
#--------------------
plt.subplot(236)
plt.hist(features_train_scaled['v227'], color='olive')
plt.xticks(fontsize=size1)
plt.yticks(fontsize=size1)
plt.title('Histogram of v227',fontsize=size0)

plt.savefig('hist.svg')
plt.show


#%%
X_train=features_train
X_test=features_test
y_train=response_train

#%%
X_train=features_train.iloc[:,feature_imp_lr]
X_test=features_test.iloc[:,feature_imp_lr]
y_train=response_train



#%% knn classifier
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(metric='minkowski', p=1, n_jobs=-1)
gs_knn=GridSearchCV(estimator=knn,
                      param_grid=[{'n_neighbors': np.arange(30,51),
                                   'algorithm': ['auto', 'ball_tree'],
                                   #'p': np.arange(1,4)
                                   }],
                      cv=10,
                      scoring='roc_auc',
                      n_jobs=-1)

X_train=features_train_scaled
X_test=features_test_scaled
y_train=response_train

%time gs_knn.fit(X_train, y_train)
gs_knn.predict_proba(X_test)
gs_knn.best_params_ 
#{'algorithm': 'ball_tree', 'n_neighbors': 48}
gs_knn.best_score_



#%% ###########################################################################
#Logistic Regression
#####################
#=========================================
tuning logistic regression
#==========================
lr = LogisticRegression(solver='saga')
gs_lr = GridSearchCV(estimator=lr,
                        param_grid=[{'penalty': ['l1','l2','elasticnet'],
                                     'l1_ratio': [0.5, 0.6, 0.7,0.75, 0.8, 0.9],
                                     'C': [0.0001, 0.0005,0.001,0.01, 0.05, 0.1,0.15, 0.2, 1]}],
                        cv=10,
                        scoring='roc_auc',
                        n_jobs=-1)
   
X_train=features_train.iloc[:,feature_imp_lr]
X_test=features_test.iloc[:,feature_imp_lr]
y_train=response_train
   
%time gs_lr.fit(X_train,y_train)
gs_lr.predict_proba(X_test)
gs_lr.best_params_ 
gs_lr.best_score_ 


#%% logistic regression based on the best param
X_train=features_train_scaled
X_test=features_test_scaled
y_train=response_train
  
lr = LogisticRegression(solver='saga',penalty='l1',l1_ratio=.7,C=.01)
lr_fit=lr.fit(X_train,y_train)
lr_coef=lr_fit.coef_[0,:]
lr_imp_loc=np.concatenate(np.argwhere(lr_coef!=0),axis=None)
features_names[lr_imp_loc]
lr_imp=pd.DataFrame(dict(features=features_names[lr_imp_loc], importance=abs(lr_coef[lr_imp_loc]))).sort_values('importance', ascending=False)
len(lr_imp_loc)


#%% bar chart of important features using logistic regression
nfeat=15
plt.figure(1,figsize=(2.5,5))
plt.barh(np.arange(nfeat)[::-1],lr_imp['importance'][:nfeat],color='orange')
plt.yticks(np.arange(nfeat)[::-1],lr_imp['features'][:nfeat], fontsize=10)
plt.xticks(fontsize=10)
plt.ylabel('Features',fontsize=10)
plt.xlabel('Abs. of coefficients',fontsize=10)
plt.title('Logistic Regression',fontsize=13)
plt.savefig('lr_imp_bar.svg')
plt.show

#%% Naive Bayes classifier
X_train=features_train_scaled.iloc[:,feature_imp_inter]
X_test=features_test_scaled.iloc[:,feature_imp_inter]
y_train=response_train

gnb = GaussianNB()
gnb_fit=gnb.fit(X_train,y_train)
gnb_pred=gnb.predict(X_test)
gnb_proba=gnb.predict_proba(X_test)

#%% Random Forest Classifier

#=====================================
#tunning random forest
#=====================================
rf = RandomForestClassifier(n_estimators=175)
gs_rfc = GridSearchCV(estimator=rf,
                   param_grid=[{'ccp_alpha': [0.005, 0.01, 0.015, 0.020],
                                'max_features': ['sqrt'],
                                'criterion': ['gini']}],
                   cv=10,
                   scoring='roc_auc',
                   n_jobs=-1)

X_train=features_train_scaled#.iloc[:,feature_imp_inter]
X_test=features_test_scaled#.iloc[:,feature_imp_inter]
y_train=response_train

%time gs_rfc.fit(X_train, y_train)
 
gs_rfc.best_params_ 
gs_rfc.best_score_ 


#%% important feature selection
X_train=features_train_scaled#.iloc[:,feature_imp_inter]
X_test=features_test_scaled#.iloc[:,feature_imp_inter]
y_train=response_train

rfc=RandomForestClassifier(n_estimators=175, ccp_alpha=0.015, criterion='gini', max_features='sqrt')
rfc_fit=rfc.fit(X_train, y_train)
rfc_imp= pd.DataFrame({'features': X_train.columns, 'importance': rfc.feature_importances_}).sort_values('importance',ascending=False)
rfc_nftr=sum((rfc_imp['importance']!=0))
rfc_features_imp=rfc_imp['features'][:rfc_nftr]
rfc_imp_loc=np.argwhere(rfc.feature_importances_!=0)

#%% Plot the feature importances of random forest
nfeat=15
plt.figure(2,figsize=(2.5,5))
plt.barh(np.arange(nfeat)[::-1],rfc_imp['importance'][:nfeat],color='olive')
plt.yticks(np.arange(nfeat)[::-1],rfc_imp['features'][:nfeat], fontsize=10)
plt.xticks(fontsize=10)
plt.ylabel('Features',fontsize=10)
plt.xlabel('Importance of Features',fontsize=10)
plt.title('Random Forest',fontsize=13)
plt.savefig('rf_imp_bar.svg')
plt.show
 

#%% Support Vector Machine
#====================================
#tuning SVM
#====================================
svm = SVC(shrinking=True, probability=True)
gs_svm = GridSearchCV(estimator=svm,
                      param_grid=[{'C': [0.001, 0.1, 1, 10],
                                   'kernel': ['rbf', 'poly','linear']}],
                      cv=10,
                      scoring='roc_auc',
                      n_jobs=-1)

X_train=features_train_scaled
X_test=features_test_scaled
y_train=response_train

%time gs_svm.fit(X_train, y_train)
gs_svm.predict_proba(X_test)
gs_svm.best_params_ 
{'C': 10, 'kernel': 'linear'}
gs_svm.best_score_



#%%  Gradient Boosting
#====================================
#tuning gradine boosting
#====================================

from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=175)
gs_gbc = GridSearchCV(estimator=gbc,
                      param_grid=[{'learning_rate': [0.01,0.05,0.1,0.15, 1],
                                   'ccp_alpha': [.005, .01, .015, .1],
                                   'max_features': np.arange(20,31)}],
                      cv=10,
                      scoring='roc_auc',
                      n_jobs=-1)

X_train=features_train_scaled
X_test=features_test_scaled
y_train=response_train

%time gs_gbc.fit(X_train, y_train)
gs_gbc.predict_proba(X_test)
gs_gbc.best_params_ 
gs_gbc.best_score_


#%%  Gradient boosting
X_train=features_train_scaled
X_test=features_test_scaled
y_train=response_train

gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.01 , max_features=24, ccp_alpha=.001 )
gbc_fit=gbc.fit(X_train,y_train)
gbc_imp_loc=np.argwhere(gbc_fit.feature_importances_!=0)
gbc_nftr=len(gbc_imp_loc);gbc_nftr
gbc_imp= pd.DataFrame({'features': X_train.columns, 'importance': gbc_fit.feature_importances_}).sort_values('importance',ascending=False)
gbc_features_imp=gbc_imp['features'][:gbc_nftr]


#%% Plot the feature importances of gradient boosting
nfeat=15
plt.figure(3,figsize=(2.5,5))
plt.barh(np.arange(nfeat)[::-1],gbc_imp['importance'][:nfeat],color='steelblue')
plt.yticks(np.arange(nfeat)[::-1],gbc_imp['features'][:nfeat], fontsize=10)
plt.xticks(fontsize=10)
plt.ylabel('Features',fontsize=10)
plt.xlabel('Importance of Features',fontsize=10)
plt.title('Gradient Boosting',fontsize=13)
plt.savefig('gb_imp_bar.svg')
plt.show



#%%
%%selection of feature importance
feature_imp_all=pd.DataFrame(rfc_imp_loc,lr_imp_loc,gbc_imp_loc, columns=['rfc','lr','gbc'])
feature_imp_all=pd.concat([rfc_imp_loc,lr_imp_loc,gbc_imp_loc], ignore_index=True, axis=1)
rfc_imp_loc+ gbc_imp_loc
feature_imp_union=np.unique(np.concatenate((rfc_imp_loc, lr_imp_loc, gbc_imp_loc), axis=None))
np.concatenate((rfc_imp_loc), axis=None)
feature_imp_inter=list(set.intersection(set(np.concatenate((rfc_imp_loc), axis=None)),set(np.concatenate((lr_imp_loc), axis=None)),
                set(np.concatenate((gbc_imp_loc), axis=None)) ))
features_names[feature_imp_inter]


#%% Naive Bayes classifier
X_train=features_train_scaled.iloc[:,feature_imp_inter]
X_test=features_test_scaled.iloc[:,feature_imp_inter]
y_train=response_train

gnb = GaussianNB()
gnb_fit=gnb.fit(X_train,y_train)
gnb_pred=gnb.predict(X_test)
gnb_proba=gnb.predict_proba(X_test)


#%%


from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras import regularizers

#%%

'''
# A non - regularized version, but the structure may works for our data

model = Sequential([Dense(32, activation='relu', input_shape=(10,)),
                    Dense(32, activation='relu'),
                    Dense(1, activation='sigmoid'),])

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(X_train, Y_train,
                 batch_size=32, epochs=100,
                 validation_data=(X_val, Y_val))

model.evaluate(X_test, Y_test)[1]

# plot loss
plt.fig()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# plot accuracy
plt.fig()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

'''
#%% Regularized NN, use more neurons with l1 regularization and dropout
model_3 = Sequential([Dense(64, activation='relu',
                            kernel_regularizer=regularizers.l2(0.01),
                            input_shape=(15,)),

                      Dense(32, activation='relu',
                            kernel_regularizer=regularizers.l2(0.01)),

                      Dense(32, activation='relu',
                            kernel_regularizer=regularizers.l2(0.01)),

                      Dense(1, activation='sigmoid',
                            kernel_regularizer=regularizers.l2(0.01)),])

model_3.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

X_train=features_train_scaled.iloc[:,feature_imp_inter]
X_test=features_test_scaled.iloc[:,feature_imp_inter]
y_train=response_train

hist_3 = model_3.fit(X_train, y_train,
                     batch_size=32, epochs=512,
                     validation_split=0.1)

 import winsound
 duration = 1000  # milliseconds
 freq = 440  # Hz
 winsound.Beep(freq, duration)

#%%

'''
plt.figure()
plt.plot(hist_3.history['loss'])
plt.plot(hist_3.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.savefig('feature_imp1.png')
plt.show()
'''


plt.figure()
plt.plot(hist_3.history['accuracy'])
plt.plot(hist_3.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.savefig('accuracy_NN.png')
plt.show()




#%% scenarios for important features

lr_features=set(lr_imp_loc)
rf_features=set(np.concatenate(rfc_imp_loc,axis=None))
gb_features=set(np.concatenate(gbc_imp_loc,axis=None))

#scenario1 based on joint of all thress lr, rf, gb
scn1=(lr_features & rf_features & gb_features);len(scn1)

#scenario1 based on joint of all thress lr, rf, gb
scn2=(lr_features | rf_features | gb_features);len(scn2)





#%% stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
 
# this part needs to substute in best tuned estimators
estimators = [('rf', RandomForestClassifier(n_estimators=175, ccp_alpha=0.01, criterion='gini')),
               #('lr', LogisticRegression(solver='saga',penalty='l1',l1_ratio=.7,C=.001)),
               #('svm', SVC(C=10,kernel='linear',shrinking=True, probability=True)),
               #('gb', GradientBoostingClassifier(n_estimators=175, learning_rate=0.075, max_features=5, max_depth=1)),
               #('gnBayes', GaussianNB()),
               #('knn', KNeighborsClassifier(metric='minkowski', algorithm='ball_tree', n_neighbors=48, p=1, n_jobs=-1)),
               #()
               ]
 
stacking_clf = StackingClassifier(estimators=estimators,
                                   final_estimator=LogisticRegression(),
                                   #passthrough=False,
                                   cv=10,
                                   #stack_method=['auto', 'predict_proba'], 
                                   n_jobs=-1)
 
gs_stacking = GridSearchCV(estimator=stacking_clf,
                            param_grid=[{'passthrough': [False], 'stack_method': ['auto', 'predict_proba']}],
                            cv=10,
                            scoring='roc_auc',
                            n_jobs=-1)
 

#X_train=features_train_scaled.loc[:,rfc_features_imp]
#X_test=features_test_scaled.loc[:,rfc_features_imp]
#X_train=features_train_scaled.iloc[:,feature_imp_union]
#X_test=features_test_scaled.iloc[:,feature_imp_union]
X_train=features_train_scaled.iloc[:,feature_imp_union]
X_test=features_test_scaled.iloc[:,feature_imp_union]
y_train=response_train


%time gs_stacking.fit(X_train, y_train)
gs_stacking.best_params_ 
gs_stacking.best_score_
stacking_proba=gs_stacking.predict_proba(X_test)


import winsound
duration = 1000  # milliseconds
freq = 440  # Hz
winsound.Beep(freq, duration)
#%%
stacking_clf = StackingClassifier(estimators=estimators,
                                   final_estimator=LogisticRegression(),
                                   passthrough=False,
                                   stack_method='auto', 
                                   n_jobs=-1)

X_train=features_train_scaled.iloc[:,feature_imp_union]
X_test=features_test_scaled.iloc[:,feature_imp_union]
y_train=response_train

stacking_fit=stacking_clf.fit(X_train, y_train)
stacking_proba=stacking_clf.predict_proba(X_test)
pd.DataFrame(stacking_proba).to_csv('stacking_proba.csv')

#%%
 import winsound
 duration = 1000  # milliseconds
 freq = 440  # Hz
 winsound.Beep(freq, duration)
# =============================================================================

















