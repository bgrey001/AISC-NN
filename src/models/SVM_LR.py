#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 18:35:11 2023
@author: benedict

Script for modelling the AIS data using a support vector machine classifier

"""
# =============================================================================
# dependencies
# =============================================================================
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pycm import ConfusionMatrix

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import  accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

sns.set_style("darkgrid") 
np.random.seed(10)


# =============================================================================
# data inspection
# =============================================================================
"""
Function to encapsulate data investigation and visualisation
@params:
    train_data
    test_data
"""
def inspect_data(train_data, test_data):
    train_data.info() # 23 features, all int64 data type, no nulls
    train_data.describe() 
    # observe data distributions for each feature
    train_data.hist(bins=50, figsize=(50,40)) 
    plt.show()
    # generate correlation matrix and see the strongest correlations with target
    corr_mat = train_data.corr()
    corr_mat['Y'].sort_values(ascending=False)
    # heatmap
    sns.heatmap(corr_mat)
    # class imbalance
    print(f"Train data target distribution:\n{train_data['Y'].value_counts(normalize=True)}")
    print(f"Test data target distribution:\n{test_data['Y'].value_counts(normalize=True)}")

  
# =============================================================================
# preprocessing
# =============================================================================
"""
Function to handle the preprocessing of the data
@params: 
    train_data
    test_data
@returns:
    X_train 
    y_train
    X_test
    y_test
"""
def preprocessing(train_data, test_data):
    # numerical pipeline, only imputing in this instance but could be used for further preprocessing steps
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )
    # get targets and features
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1].values
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1].values
    # apply column transformer on train and test
    X_train = numeric_transformer.fit_transform(X_train)
    X_test = numeric_transformer.transform(X_test)
    return X_train, y_train, X_test, y_test
    
    
"""
Is this necessary?
"""
def feature_selection(X_train, X_test, y_train, y_test, k=2, verbose=True):
    return

"""
Function to calculate metrics for evaluation of models
@params: 
    y_test - ground truth target values
    y_preds - predicted target values
    y_pp - predicted probabilities
@returns: list of metrics
"""
def compute_metrics(y_test, y_preds, y_pp):
    # metrics computed using regular threshold (0.5, sci-kit learn default for binary classification)
    acc = accuracy_score(y_test, y_preds)
    prec = precision_score(y_test, y_preds, average=None)
    rec = recall_score(y_test, y_preds, average=None)
    f1 = f1_score(y_test, y_preds, average=None)
    roc_auc = roc_auc_score(y_test, y_pp, multi_class='ovr')
    # fpr, tpr, _ = roc_curve(y_test, y_pp)
    # precision, recall, _ = precision_recall_curve(y_test, y_pp)
    # ap = average_precision_score(y_test, y_pp, average='macro', mul)
    return [acc, prec, rec, f1, roc_auc], f1


if __name__ == "__main__":
    # load data from csv
    df = pd.read_csv('../../data/csv/fe_pd.csv')    
    # split data
    train_data, test_data = train_test_split(df, test_size=0.25, stratify=df['Y'])
    # inpsect data
    inspect_data(train_data, test_data)
    # preprocess the data
    X_train, y_train, X_test, y_test = preprocessing(train_data, test_data)
    
    # =============================================================================
    # SVM
    # =============================================================================
    # init model
    hyperparams =  {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1, 1]
        }
    
    model = SVC(C=1.0, probability=True, verbose=0)
    
    # init grid search obj
    grid_cv = GridSearchCV(estimator=model, param_grid=hyperparams, scoring='accuracy', cv=2, n_jobs=-1, refit=True, verbose=1)
    
    start_time = datetime.now()
    # run grid search
    # grid_fit = grid_cv.fit(X_train, y_train)
    model.fit(X_train, y_train)
    end_time = datetime.now()
    print('TIME ELAPSED: ', end_time - start_time)
    
    y_preds = model.predict(X_test)
    y_pp = model.predict_proba(X_test)
    
    results_df = pd.DataFrame(columns=['Model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc'])    
    metrics, f1 = compute_metrics(y_test, y_preds, y_pp)
    metrics.insert(0, 'SVM')
    results_df.loc[len(results_df)] = metrics
    
    
    confmat = ConfusionMatrix(actual_vector=y_test, predict_vector=y_preds)
    confmat.plot(cmap=plt.cm.Reds,number_label=True,plot_lib="matplotlib")
    plt.savefig('../../plots/ML/SVM_cm.png', dpi=300)  
    confmat.print_matrix()
    confmat.stat(summary=True)
    # =============================================================================
    # logistic regression
    # =============================================================================
    start_time = datetime.now()
    linear_model = LogisticRegression(max_iter=100000)
    linear_model.fit(X_train, y_train)
    end_time = datetime.now()
    print('TIME ELAPSED: ', end_time - start_time)
    # y_preds2 = linear_model.predict(X_test)
    
    confmat2 = ConfusionMatrix(actual_vector=y_test.astype(int), predict_vector=y_preds2.astype(int))
    confmat2.plot(cmap=plt.cm.Greens,number_label=True,plot_lib="matplotlib")
    plt.savefig('../../plots/ML/LR_cm.png', dpi=300)    
    confmat2.print_matrix()
    confmat2.stat(summary=True)
    









