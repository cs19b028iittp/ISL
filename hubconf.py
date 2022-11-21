
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
  lr_param_grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
  # refer to sklearn documentation on grid search and logistic regression
  # write your code here...
  return lr_param_grid

def get_paramgrid_rf():
  # you need to return parameter grid dictionary for use in grid search cv
  # n_estimators: 1, 10, 100
  # criterion: gini, entropy
  # maximum depth: 1, 10, None  
  rf_param_grid = {
        'max_depth': [1, 10, None],
        'n_estimators': [1, 10, 100],
        'criterion': ['gini', 'entropy']
    }
  # refer to sklearn documentation on grid search and random forest classifier
  # write your code here...
  return rf_param_grid

def perform_gridsearch_cv_multimetric(model1=None, param_grid=None, cv=5, X=None, y=None, metrics=['accuracy','roc_auc']):
  
  # you need to invoke sklearn grid search cv function
  # refer to sklearn documentation
  # the cv parameter can change, ie number of folds  
  
  # metrics = [] the evaluation program can change what metrics to choose
  
  grid_search_cv = GridSearchCV(model1, param_grid, cv=cv)
  grid_search_cv.fit(X, y)
  params = grid_search_cv.best_params_
  acc, prec, rec, f1, auc = 0, 0, 0, 0, 0
  if 'criterion' in params.keys():
    rfc1 = RandomForestClassifier(random_state=42, n_estimators=params['n_estimators'], max_depth=params['max_depth'], criterion=params['criterion'])
    rfc1.fit(X,y)
    acc, prec, rec, f1, auc = get_metrics(rfc1, X, y)
  else:
    lg1 = LogisticRegression(C=params['C'], penalty=params['penalty'],solver = "liblinear")
    lg1.fit(X, y)
    acc, prec, rec, f1, auc = get_metrics(lg1, X, y)
  # create a grid search cv object
  # fit the object on X and y input above
  # write your code here...
  
  # metric of choice will be asked here, refer to the-scoring-parameter-defining-model-evaluation-rules of sklearn documentation
  
  # refer to cv_results_ dictonary
  # return top 1 score for each of the metrics given, in the order given in metrics=... list
  
  top1_scores = []
  for k in metrics:
      if k == 'accuracy':
          top1_scores.append(acc)
      elif k == 'recall':
          top1_scores.append(rec)
      elif k == 'roc_auc':
          top1_scores.append(auc)
      elif k == 'precision':
          top1_scores.append(prec)
      else:
          top1_scores.append(f1)
        

  return top1_scores 


class MyNN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(MyNN,self).__init__()
    
    self.fc_encoder = nn.Linear(inp_dim,hid_dim) 
    self.fc_decoder = nn.Linear(hid_dim,inp_dim) 
    self.fc_classifier = nn.Linear(hid_dim,num_classes) 
    
    self.relu = nn.ReLU() #write your code - relu object
    self.softmax = nn.Softmax() #write your code - softmax object
    
  def forward(self,x):
    x = torch.flatten(x) # write your code - flatten x
    x = torch.nn.functional.normalize(x, p=2.0, dim=0)
    x_enc = self.fc_encoder(x)
    x_enc = self.relu(x_enc)
    
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax(y_pred)
    
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
  def loss_fn(self,x,yground,y_pred,xencdec):
    lc1 = -(torch.nn.functional.one_hot(yground,num_classes=y_pred.shape[-1])*torch.log(y_pred)) # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    lc1=torch.mean(lc1)
    lc2 = torch.mean((x - xencdec)**2)
    return lc1+lc2
    
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  X,y = load_digits(return_X_y=True)
  X_tensor=torch.tensor(X)
  y_tensor=torch.tensor(y)
  return X_tensor,y_tensor

def get_loss_on_single_point(mynn,x0,y0):
  y_pred, xencdec = mynn(x0)
  lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
  return lossval

def train_combined_encdec_predictor(mynn,X,y, epochs=11):
  # X, y are provided as tensor
  # perform training on the entire data set (no batches etc.)
  # for each epoch, update weights
  
  optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  
  for i in range(epochs):
    for j in range(X.shape[0]):
      try:
        optimizer.zero_grad()
        ypred, Xencdec = mynn(X[j])
        lval = mynn.loss_fn(X[j],y,ypred,Xencdec)
        lval.backward()
        optimizer.step()
      except:
        pass
  return mynn
