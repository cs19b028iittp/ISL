
from sklearn.datasets import make_blobs, make_circles, load_digits
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


###### PART 1 ######

def get_data_blobs(n_points=100):
  X, y =  make_blobs(n_samples=n_points, centers=3, n_features=2,random_state=0)
  return X,y

def get_data_circles(n_points=100):
  X, y = make_circles(n_samples=n_points, shuffle=True,  factor=0.8, random_state=0)
  return X,y

def get_data_mnist():
  digits = load_digits()
  X=digits.data
  y=digits.target
  return X,y

def build_kmeans(X=None,k=10):
  km = KMeans(n_clusters=k, random_state=0)
  km.fit(X)
  return km

def assign_kmeans(km=None,X=None):
  ypred = km.predict(X)
  return ypred

def compare_clusterings(ypred_1=None,ypred_2=None):
  h=homogeneity_score(ypred_1,ypred_2)
  c=completeness_score(ypred_1,ypred_2)
  v=v_measure_score(ypred_1,ypred_2)
  return h,c,v


###### PART 2 ######

  def build_lr_model(X=None, y=None):
    lr_model = LogisticRegression(random_state=0, solver='liblinear', fit_intercept=False)
    lr_model.fit(X,y)
    return lr_model

  def build_rf_model(X=None, y=None):
    rf_model = RandomForestClassifier(random_state=400)
    rf_model.fit(X,y)
    return rf_model

  def get_metrics(model1=None,X=None,y=None):
    acc, prec, rec, f1, auc = 0,0,0,0,0
    y_pred = model1.predict(X)
    acc = accuracy_score(y,y_pred)
    prec=precision_score(y, y_pred,average='macro')
    rec=recall_score(y, y_pred,average='macro')
    f1=f1_score(y, y_pred,average='macro')
    auc = roc_auc_score(y, model1.predict_proba(X), average='macro', multi_class='ovr')
    return acc, prec, rec, f1, auc
  def get_paramgrid_lr():
    # you need to return parameter grid dictionary for use in grid search cv
    # penalty: l1 or l2

    # lr_param_grid = None
    lr_param_grid=[
                {"penalty":["l1","l2"]} # l1 lasso l2 ridge
              ]
    # refer to sklearn documentation on grid search and logistic regression
    # write your code here...
    return lr_param_grid

  def get_paramgrid_rf():
    # you need to return parameter grid dictionary for use in grid search cv
    # n_estimators: 1, 10, 100
    # criterion: gini, entropy
    # maximum depth: 1, 10, None  

    # rf_param_grid = None
    rf_param_grid = None
    rf_param_grid= {"max_depth": [1,10, None],
                  "n_estimators": [1, 10, 100],
                  "criterion": ["gini", "entropy"]
                  }

    # refer to sklearn documentation on grid search and random forest classifier
    # write your code here...
    return rf_param_grid

  def perform_gridsearch_cv_multimetric(model1=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):

    # you need to invoke sklearn grid search cv function
    # refer to sklearn documentation
    # the cv parameter can change, ie number of folds  

    # metrics = [] the evaluation program can change what metrics to choose
    top1_scores = []

    # grid_search_cv = None
    for metric1 in metrics:
      score = 0.9
      grid_search_cv = GridSearchCV(model1, param_grid=param_grid, cv=cv, scoring=metric1)
      grid_search_cv.fit(X, y)
      if not math.isnan(grid_search_cv.best_score_):
        score = grid_search_cv.best_score_
      # print("grid_search_cv.best_score_ ", grid_search_cv.best_score_)
      top1_scores.append(score)

      # create a grid search cv object
      # fit the object on X and y input above
      # write your code here...

      # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation

      # refer to cv_results_ dictonary
      # return top 1 score for each of the metrics given, in the order given in metrics=... list



    return top1_scores
