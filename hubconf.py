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
from torch import nn
import torch.nn as nn
import torch.nn.functional as F
class MyNN(nn.Module):
  def __init__(self,inp_dim=64,hid_dim=13,num_classes=10):
    super(MyNN,self).__init__()
    
    self.fc_encoder = nn.Linear(inp_dim,hid_dim) # write your code inp_dim to hid_dim mapper
    self.fc_decoder = nn.Linear(hid_dim,inp_dim) # write your code hid_dim to inp_dim mapper
    self.fc_classifier = nn.Linear(hid_dim,num_classes) # write your code to map hid_dim to num_classes
    
    self.relu = nn.ReLU #write your code - relu object
    self.softmax = nn.Softmax #write your code - softmax object
    
  def forward(self,x):
    x = torch.flatten(x) # write your code - flatten x
    x_enc = self.fc_encoder(x)
    x_enc = self.relu()(x_enc)
    
    y_pred = self.fc_classifier(x_enc)
    y_pred = self.softmax()(y_pred)
    
    x_dec = self.fc_decoder(x_enc)
    
    return y_pred, x_dec
  
  # This a multi component loss function - lc1 for class prediction loss and lc2 for auto-encoding loss
  def loss_fn(self,x,yground,y_pred,xencdec):
    
    # class prediction loss
    # yground needs to be one hot encoded - write your code
    yground = F.one_hot(yground,num_classes=y_pred.shape[0])
    lc1 = -torch.mean(torch.mul(yground,torch.log(y_pred+0.001))) # write your code for cross entropy between yground and y_pred, advised to use torch.mean()
    
    # auto encoding loss
    lc2 = torch.mean((x - xencdec)**2)
    
    lval = lc1 + lc2
    
    return lval
    
def get_mynn(inp_dim=64,hid_dim=13,num_classes=10):
  mynn = MyNN(inp_dim,hid_dim,num_classes)
  mynn.double()
  return mynn

def get_mnist_tensor():
  # download sklearn mnist
  # convert to tensor
  npX,npy = get_data_mnist()
  X, y = torch.from_numpy(npX),torch.from_numpy(npy)
  # write your code
  return X,y

def get_loss_on_single_point(mynn,x0,y0):
  y_pred, xencdec = mynn(x0)
  lossval = mynn.loss_fn(x0,y0,y_pred,xencdec)
  print(lossval)
  # the lossval should have grad_fn attribute set
  return lossval

def train_combined_encdec_predictor(mynn,X,y, epochs=11):
  # X, y are provided as tensor
  # perform training on the entire data set (no batches etc.)
  # for each epoch, update weights
  
  optimizer = optim.SGD(mynn.parameters(), lr=0.01)
  
  for i in range(epochs):
    optimizer.zero_grad()
    ypred, Xencdec = mynn(X)
    lval = mynn.loss_fn(X,y,ypred,Xencdec)
    lval.backward()
    optimizer.step()
    
  return mynn
