
# coding: utf-8

# ### Note 
# You will have to run this notebook six times:
# 1. Support with Tfidf
# 2. Support with Embeddings
# 3. Opposition with Tfidf
# 4. Opposition with Embeddings
# 5. Both with Tfidf
# 6. Both with Embeddings
# 
# 
# The variables to pay attention too are `feature` and `key`
# 
# `key` can be support , oppostion or both
# `feature` can be tfidf or embedding
# 
# 
# if  `key` is both then set  `both` in  `Data(testset[testset['data_type'] == 'train'], feature=feature ,both=?)` to `True`
# 
# e.g Data(testset[testset['data_type'] == 'train'], feature=feature ,both=True)
# 
# Else if `key` is not both, thus `support` or `opposition` then set `both` to `False`
# 
# Data(testset[testset['data_type'] == 'train'], feature=feature ,both=True)

# In[1]:


import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import ast
from joblib import dump, load
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from typing import List, Set, Dict, Tuple
import random
import torch.nn.functional as F
from torch.nn import (
    Sequential as Seq,
    Linear as Lin,
    ReLU,
    BatchNorm1d,
    AvgPool1d,
    Sigmoid,
    Conv1d,
)
import wandb

from deepsetmodel import *



# ### Dataset
# 
# Here i load a paired datatset.
# 
# Unlike the other (unpaird) datatset which contains individual briefs. 
# 
# Each row of the paired datast contains two set of briefs. Where each set corresponds to either oppostion and support.
# 
# 

# In[2]:


DEVICENUMBER = 1

if torch.cuda.is_available():
    device = torch.device(F"cuda:{DEVICENUMBER}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

device = torch.device("cpu")
torch.cuda.empty_cache() 


PAIRED_PATH = '../dataset/paired_testset.csv' #'../summaries/summarized_paired_testset.csv' # '../dataset/paired_testset.csv'

testset = pd.read_csv(PAIRED_PATH, sep=',',index_col=0)

testset = testset.loc[testset['data_type'] == 'train']

# randomly set 20% of the data to test
testset['data_type'] = testset['data_type'].apply(lambda x: "test" if random.random() < 0.2 else "train")
testset[testset['data_type'] == 'test'].shape , testset[testset['data_type'] == 'train'].shape


# ### DataLoader *
# 
# Just like the LLM Notebook you will have to run the cells below multiple times to run different experiments.
# 
# As of the time this note was written,  If you want to specify the input type ("tfidf" or "embedding"), then set feature variable below.
# 
# The Data class returns a pytorch dataset object.
# 
# 
# 
# 
# 

# ##### Data class definition
# 
# Below is the signature of the Data class:
# 
# <font color="blue">def __init__(self,df,feature = 'tfidf', getEmbeddings = sentence_model.encode ,support_pipe = '../pipes/support-tfidf.joblib', opposition_pipe = '../pipes/oppose-tfidf.joblib', both =False, both_pipe = '../pipes/both-tfidf.joblib')</font>
# 
# `self.feature`: This specifies the feature engineering technique to use on the dataset. 
#                 There are only 2 feature types implemented: ["tfidf", "embeddings"]
# 
# 
# `self.getEmbeddings`: This refers to the embedding function that will be used on the text. You can pick use your own embedding function. You don't have to specify an embedding function the default is already specified for you. 
# 
# `self.both`: When set to True a dataset where the supporting and opposing briefs are joined (Unioned) together to from one set of briefs.
#              When Set to False a dataset of disjointed supporting and oppossing breifs are joined. 
#              You have to make sure that if you set this to true, then you have to set the configuration for the model to accept datasets that are joined.
# 
# `self.support_pipe, self.opposition_pipe`: These are paths to pipes that convert text to tfidf vectors. Default paths are provided, so you don't have to explicitly set it.
# 
# `self.both_pipe`: This is a path that converst text to tfidf vectors. This requires particular attention, because this pipe is constructed from both support and opposition. It is does a requirement is you set `self.both` to True. Default paths are provided, so you don't have to explicitly set it.
# 
# 
# 

# ##### Examples and Usecase:
# 
# Example 1.
# <font color="blue">train_data = Data(testset[testset['data_type'] == 'train'], feature='embedding' ,both=True)</font>:
# 
# returns an embedding dataset where support and oppositions are combined.
# 
# Example 2.
# <font color="blue"> train_data = Data(testset[testset['data_type'] == 'test'], feature='tfidf' ,both=True) </font>:
# 
# returns a tfidf dataset where where support and opposiontions are seperated.
# 
# Example 3.
# ```
# def embedding_func(brief):
#     
#     ....
# ```
# <font color="blue"> train_data = Data(testset[testset['data_type'] == 'test'], feature='embedding',both=True , getEmbeddings = embedding_func ): </font>
# 
# returns a embedding dataset where the embedding function is explicitly defined.
# 

# In[3]:


sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

class Data(Dataset):
    def __init__(self,df,feature = 'tfidf', getEmbeddings = sentence_model.encode ,support_pipe = '../pipes/support-tfidf.joblib', opposition_pipe = '../pipes/oppose-tfidf.joblib', both =False, both_pipe = '../pipes/both-tfidf.joblib'):
        self.df = df    
        supports = self.df['support'].values
        oppositions = self.df['opposition'].values
        self.folder_id = self.df['folder_id'].values
        self.y = self.df['outcome'].values 
        # convert list of stings to list of lists of stings
        supports = list(map(lambda x: ast.literal_eval(x), supports))
        oppositions = list(map(lambda x: ast.literal_eval(x), oppositions))
        self.both = both
        if self.both:
            self.combined = list(map(lambda x,y: x+y, supports,oppositions))


        self.getEmbeddings = getEmbeddings
        
        if self.both == False:
            self.max_len_brief = max(self.findMaxLen(supports),self.findMaxLen(oppositions))
        else:
            self.max_len_brief = self.findMaxLen(self.combined)

        if feature == 'tfidf':
            if self.both == False:
                support_pipe = load(support_pipe)
                opposition_pipe = load(opposition_pipe)
                getSupport = lambda x: self.stringsToTfidfs(x,support_pipe)
                getOpposition = lambda x: self.stringsToTfidfs(x,opposition_pipe)


                self.supports = list(map( getSupport, supports))
                self.oppositions = list(map( getOpposition, oppositions))

            else:
                both_pipe = load(both_pipe)
                getTfidf= lambda x: self.stringsToTfidfs(x,both_pipe)
                self.combined = list(map( getTfidf, self.combined))

        elif feature == 'embedding':
            if self.both == False:
                self.supports: list = list(map(lambda x: self.stringsToEmbeddings(x), supports))
                self.oppositions: list = list(map(lambda x: self.stringsToEmbeddings(x), oppositions))
            else:
                self.combined: list = list(map(lambda x: self.stringsToEmbeddings(x), self.combined))

        
    def __len__(self):
        if self.both == False:
            return len(self.supports)
        else:
            return len(self.combined)
    
    def __getitem__(self, idx):
        y = 1.0 if self.y[idx] == 'grant' else 0.0

        if hasattr(self, 'combined') and self.both == True:
            return self.combined[idx] , y , self.folder_id[idx]
        else:
            return self.supports[idx] , self.oppositions[idx] , y , self.folder_id[idx]
        
    def findMaxLen(self,x):
        max_len = 0
        for i in range(len(x)):
            row = x[i]
            if len(row) > max_len:
                max_len = len(row)
        return max_len

    def stringsToTfidfs(self,briefs: List[str],pipe):
        tfidfs = torch.tensor(pipe.transform(briefs).toarray(),dtype=torch.float32)

        return self.padFeatures(tfidfs)
    

    
    def stringsToEmbeddings(self,briefs: List[str]):
        embeddings =  torch.tensor(self.getEmbeddings(briefs),dtype=torch.float32)
        return self.padFeatures(embeddings)
    
    def padFeatures(self,features: List[torch.tensor]):
        num_padding = self.max_len_brief - features.shape[0]
        padding = nn.ConstantPad2d((0, 0, 0, num_padding), 0)
        features = padding(features)
        features = features.T
        return features
    

feature = "tfidf"

train_data = Data(testset[testset['data_type'] == 'train'], feature=feature ,both=True)
test_data = Data(testset[testset['data_type'] == 'test'], feature=feature ,both=True)

batch_size = 3

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# warning about pickle version *Vectorizer from version 1.3.2 when using version 1.4.0


# ### Model creation *
# 
# In the cell below is where the size of the model is defined.
# Here you have to ability to increase or decreace the number of hidden units per layer.
# 
# Just like the Dataloader tab above you have to run this cell once per experiment.
# 
# You can only select one out of three types of models ["support", "opposition","both"]
# 
# specify your choice in the variable key
# 
# Example
# ``` key = "both" ```
# 
# 
# 
# 
# 
# ##### Notes to myself:
# 
# The performance of the DeepSet model maybe hampered because of the small latent space.
# The machie i am using cannot handle a larger latent space. Once you get a larger GPU memory retry this scipt with a larger latent space
# 
# 
# 
# Ideas on reducing the load on GPUs:
# 
# https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html
# 
# When you have time learn this:
# 
# https://docs.wandb.ai/guides/model_registry/walkthrough

# In[4]:



# TFIDF is quiet big so i may have to reduce the hiden layers width
# the latent space has the be atleast the size of the input

models = {}
optimizers = {}

key = "both"

if key == "support":
    input_size = train_data.supports[0].shape[0]
elif key == "opposition":
    input_size = train_data.oppositions[0].shape[0]
else: #key == "both"    
    input_size = train_data.combined[0].shape[0]


max_len_brief = train_data.max_len_brief

hidden1 = int(input_size /5)
hidden2 = int(hidden1 / 4)
hidden3 = int(hidden2 / 3)
classify1 = int(hidden3 /2)

models[key] = DeepSets(input_size, max_len_brief , hidden1, hidden2, hidden3, classify1).to(device)

latent_size = int(input_size / 10)
hidden_size = latent_size
output_size =  1


## what does Bachnorm and conv1d work?
lr = 1e-4   
optimizers[key] = torch.optim.Adam(models[key].parameters(), lr=lr)
#optimizers["opposition"] = torch.optim.Adam(models["opposition"].parameters(), lr=1e-4)
#optimizers["both"] = torch.optim.Adam(models["both"].parameters(), lr=1e-2)


# ### Functions
# 
# These are functions for  training, and testing

# In[5]:


from tqdm.notebook import tqdm

@torch.no_grad()
def test(model, loader, total, batch_size, leave=False , datatype='support', loss_fn= nn.BCELoss()):
    
    model.eval()

    sum_loss = 0.0
    sum_acc = 0.0

    t = tqdm(enumerate(loader), total=total /batch_size, leave=leave)

    csv = {'folder':[],'prediction':[], 'score':[], 'truth':[]}

    for i, data in t:

        if datatype != "both":
            supports, oppositions, y , folder_id = data
            supports = supports.to(device)
            oppositions = oppositions.to(device)
        else:
            combined, y , folder_id = data
            combined = combined.to(device)

        y = y.float()
        y = y.reshape(-1,1)
        y = y.to(device)

        if datatype == 'support':
            outputs= model(supports)
        elif datatype == 'opposition':
            outputs= model(oppositions)
        elif datatype == 'both':
            outputs= model(combined)

        loss = loss_fn(outputs, y)
        predictions = (outputs > 0.5)
        acc = (predictions == y).sum().item()
        sum_acc += acc
        avg_acc =  acc /batch_size
        
        sum_loss += loss.item()

        t.set_description(f"batch_loss_{datatype}: {loss.item():.4f} \t| sum_loss_{datatype}: {sum_loss:.4f}\n batch_accuracy_{datatype}: {avg_acc:.4f}")
        
        t.refresh()

        csv['folder'].extend(folder_id)
        csv['prediction'].extend(predictions.cpu().numpy().flatten())
        csv['score'].extend(outputs.cpu().numpy().flatten())
        csv['truth'].extend(y.cpu().numpy().flatten())

        
    # what is the (i+1) for?
        
    return sum_loss  / len(loader.dataset) , sum_acc / len(loader.dataset) , pd.DataFrame(csv)


def train(model, optimizer, loader, total, batch_size, leave=False, datatype='support', loss_fn= nn.BCELoss()):
    model.train()

    sum_loss = 0.0
    t = tqdm(enumerate(loader), total=total /batch_size, leave=leave)
    for i, data in t:

        if key != "both":
                
            supports, oppositions, y , _ = data
            supports = supports.to(device)
            oppositions = oppositions.to(device)

        else:
            combined, y , _ = data
            combined = combined.to(device)

        y = y.float()
        y = y.reshape(-1,1)
        y = y.to(device)

        optimizer.zero_grad()

        if datatype == 'support':
            outputs= model(supports)
        elif datatype == 'opposition':
            outputs= model(oppositions)
        elif datatype == 'both':
            outputs= model(combined)
        loss = loss_fn(outputs, y)
        sum_loss += loss.item()

        #wandb.log({"batch_loss": loss.item() } )
      
        loss.backward()

        optimizer.step()

        t.set_description(f"batch_loss_{datatype}: {loss.item():.4f} \t| sum_loss_{datatype}: {sum_loss:.4f}")
        t.refresh()

    return sum_loss / len(loader.dataset)


# ### Train
# 
# The cell below is where the model is trained , tested, used for inference and saved to disk.
# 
# 

# In[6]:


import os.path as osp

n_epochs = 300
stale_epochs = 0
best_valid_acc = 0.0
patience = 100
t = tqdm(range(0, n_epochs))
  

wandb.init(
    # set the wandb project where this run will be logged
    project="DeepSets",  
    name= f"{key}s-{feature}-epochs:{n_epochs}-patience:{patience} epochs",
    
    # track hyperparameters and run metadata
    config={

    "optimizer": "AdamW",
    
    "lr": lr,

    "dataset": f"single-{key}",

    "epochs": n_epochs,

    "patience": patience,

    "architecture":"ConvolutionalDeepSets",


    "hidden1" : hidden1,

    "hidden2" : hidden2,

    "hidden3" : hidden3,

    "classify1" : classify1,


    }
)


for epoch in t:
    avg_loss = train(
        model=models[key], 
        optimizer=optimizers[key], 
        loader=train_loader, 
        total=len(train_data), 
        batch_size=batch_size, 
        leave=bool(epoch == n_epochs - 1),
        datatype=key 
    )
    
    
    valid_loss, valid_acc , csv = test(
        model=models[key],
        loader=test_loader, 
        total=len(test_data), 
        batch_size=batch_size, 
        leave=bool(epoch == n_epochs - 1),
        datatype=key
    )
    
    wandb.log({"train_loss": avg_loss, "valid_loss": valid_loss, "valid_acc": valid_acc})

    print("Epoch: {:02d}, Training Loss:   {:.4f}".format(epoch, avg_loss))
    print("           Validation Loss: {:.4f}".format(valid_loss))
    print("           Validation Accuracy: {:.4f}".format(valid_acc))

    if valid_acc > best_valid_acc:
        best_valid_acc = valid_acc
        modpath = osp.join(f"../models/DeepSets_{key}_{feature}.pth")
        print("New best model saved to:", modpath)
        torch.save(models[key].state_dict(), modpath)
        # save csv
        csv["folder"] = csv["folder"].astype(int)
        csv["prediction"] = ["grant" if x == 1 else "deny" for x in csv["prediction"]]
        csv["truth"] = ["grant" if x == 1 else "deny" for x in csv["truth"]]
        
        csv.to_csv(f"../predictions/DeepSets_{key}_{feature}_predictions.csv", index=False)

        stale_epochs = 0
    else:
        print("Stale epoch")
        stale_epochs += 1
    if stale_epochs >= patience:
        print("Early stopping after %i stale epochs" % patience)
        break

wandb.finish()


torch.cuda.empty_cache()

del models[key]


# In[16]:


total


# In[12]:


supports.shape

