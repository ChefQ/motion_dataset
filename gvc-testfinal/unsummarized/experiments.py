
# coding: utf-8

# In[19]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.svm import LinearSVC
import torch
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from pickle import dump, load
import joblib

import re


from scipy import sparse
from sentence_transformers import SentenceTransformer
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# ### Data (1)
# 
# The tab below retrieves  all the data that will be relevant for building the traditional classifiers.
# 
# set the variables  `PAIRED_PATH` and  `UNPAIRED_PATH` appropriatly

# In[20]:



# path = f'hidden_states/paired_testset_embeddings_{0}-{50}.csv'
# df = pd.read_csv(path, index_col=0)
# for i in range(50, 500, 50):
#     path = f'hidden_states/paired_testset_embeddings_{i}-{i+50}.csv'
#     temp = pd.read_csv(path)
#     df = pd.concat([df, temp])



PAIRED_PATH = './dataset/paired_testset.csv'
UNPAIRED_PATH = './dataset/testset.csv'
paired = pd.read_csv(PAIRED_PATH, index_col=0)

testset = pd.read_csv(UNPAIRED_PATH, index_col=0)


# ### Functions (2)
# 
# This tab contains the definition and configuration of the traditional classifiers.
# 
# It also contains some utility functions. The most relevant one is 
# `accuracies(trainX, trainY, testX, testY, models=models, feature_type='')`
# 
# This function takes a model aswell as pairwise data points. The function trains, saves and reports the perfomance of the model.

# In[21]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


clf1 = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)

clf2 = RandomForestClassifier(n_estimators = 1500, criterion = "entropy", oob_score = True, max_features= 1)

clf3 = KNeighborsClassifier()

clf4 = LogisticRegression(solver='liblinear')

clf5 = LinearSVC( random_state=42, tol=1e-5)

models = {
    "SGD": clf1,
    "RFT": clf2,
    "KNN" : clf3,
    "Logistic": clf4,
    "LinearSVC": clf5
}
# Create a function  that maps vector_indices to a sparse vector
# The function should not produce any errors
# The function should produce a sparse vector
# The function takes in two arguments: vector_indices and max_index

def getSingleBriefsTfidf(testset, vocab_length):
    """
    This function converts the tfidf dictionary into a sparse matrix
    Concatates all the sparse matrices into a single matrix
    Concatates the completion into a single array
    Concatates the the data_type into a single array

    return X, y, data_type , brief_type
    """
    for i in range(0, len(testset)):
        row = testset.iloc[i]
        if i == 0:
            X = map_vector_indices_to_sparse_vector(row['tfidf'], vocab_length)
            y = np.array([row['completion']])
            data_type = np.array([row['data_type']])
            brief_type = np.array([row['brief_type']])
        else:
            X = concatenate_sparse_vectors(X, map_vector_indices_to_sparse_vector(row['tfidf'], vocab_length))
            y = np.append(y, row['completion'])
            data_type = np.append(data_type, row['data_type'])
            brief_type = np.append(brief_type, row['brief_type'])

    return X, y, data_type , brief_type


def map_vector_indices_to_sparse_vector(vector_indices, max_index):
    vector = np.zeros(max_index)
    for index, value in vector_indices.items():
        vector[index] = value
    return sparse.csr_matrix(vector)
    

# Create a function that concatenates two sparse vectors
# The function should not produce any errors
# The function should produce a sparse vector
# The function takes in two arguments: vector1 and vector2
def concatenate_sparse_vectors(vector1, vector2):
    return sparse.vstack((vector1, vector2))


def accuracies(trainX, trainY, testX, testY, models=models, feature_type=''):


    for key, model in models.items():

        model.fit(trainX, trainY)

        with open(f'models/{key}-{feature_type}.pkl', 'wb') as f:  # open a text file
            dump(model, f) 

        print(f"{key}: {model.score(testX, testY)}")

def cvAcurracies(X,y, models=models):

    for key, model in models.items():

        y_pred = cross_val_predict(model, X, y , cv=10)
        print(f"{key}")
        print(classification_report(y, y_pred , target_names=["Deny", "Grant"]))
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        print(f"{key}=>tn:{tn}, fp:{fp}, fn:{fn}, tp:{tp}")



def getSingleBriefEmbeddings(brief_type, paired):
    embeddings_map = list(map( lambda x: convert_to_numpy(x) , paired[brief_type].to_list()))
    outcome_map = paired['outcome'].to_list()
    data_type_map = paired['data_type'].to_list()  
    data =  [] 
    outcomes = []
    data_types = []

    for index,embeddings in enumerate(embeddings_map):
        outcome = outcome_map[index]
        data_type = data_type_map[index]
        for embedding in embeddings: 
            data.append(embedding)
            outcomes.append(outcome)
            data_types.append(data_type)
    return np.array(data) , np.array(outcomes , dtype=str) , np.array(data_types , dtype=str)

# convert string to Torch array

def convert_to_numpy(embedding):
    return np.array(ast.literal_eval(embedding))


def getSingleBriefs(brief_type, paired=paired):
    supports_map = list(map( lambda x: ast.literal_eval(x) , paired[brief_type].to_list()))
    outcome_map = paired['outcome'].to_list()
    data_type_map = paired['data_type'].to_list()  
    data = []

    for index,documents in enumerate(supports_map):
        outcome = outcome_map[index]
        data_type = data_type_map[index]
        for document in documents: 
            data.append((document,outcome,data_type))

    return np.array(data, dtype=str)

def confusion_plot(true_labels, predicted_labels, classes, labels , normalize=False, title=None, cmap=plt.cm.Blues):

    plt.figure(figsize=(10, 10))
    matrix = confusion_matrix(true_labels, predicted_labels, labels=labels)

    sns.heatmap(matrix, cmap=cmap, annot=True,
                cbar = True, fmt=".1f",
                 xticklabels=classes["x"], 
                 yticklabels=classes["y"])
    class_names = ["No Match", "Match"]
    # Plot non-normalized confusion matri
    plt.xlabel("Predicted")
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()
    return matrix



# ### TFIDF (3)
# 
# 
# This is the TFIDF part of experiment. Here i train traditional models with TFIDF embeddings. 
# 
# The text are fed through a TFIDF Pipeline.
# 
# Under this cell, I also included confusion matries for the perfomance of these models.
# 
# However, for the scope of replicating the results, the TFIDF training and testing process ends before the note seciont that says `DONE`. 
# 
# Please after you see `DONE` move towards the `Embeddings` section.
# 
# Note to self: convert Gordon's TFIDF's to format compatible to sklearns models this site might help : https://stackoverflow.com/questions/7922487/how-to-transform-numpy-matrix-or-array-to-scipy-sparse-matrix
# 
# 

# In[22]:


# supports
support_data = getSingleBriefs('support')
oppose_data = getSingleBriefs('opposition')


# In[23]:


# support

support_x = support_data[:,0]
support_target = support_data[:,1]

train = support_data[:,2] == 'train'
test = support_data[:,2] == 'test'

support_train_x = support_x[train]
support_train_target = support_target[train]
support_test_x = support_x[test]
support_test_target = support_target[test]


# opposition

oppose_x = oppose_data[:,0]
oppose_target = oppose_data[:,1]

train = oppose_data[:,2] == 'train'
test = oppose_data[:,2] == 'test'

oppose_train_x = oppose_x[train]
oppose_train_target = oppose_target[train]
oppose_test_x = oppose_x[test]
oppose_test_target = oppose_target[test]


labels = ["grant", "deny"]
classes = {"x": labels, "y": labels}

# concatenate the two oppose_x and support_x
all_x = np.concatenate((support_train_x, oppose_train_x), axis=0)

pipe = Pipeline([('count', CountVectorizer()),('tfid', TfidfTransformer())])

transformed_pipe = pipe.fit(all_x)

joblib.dump(transformed_pipe, 'pipes/both-tfidf.joblib')


# In[26]:


pipe = Pipeline([('count', CountVectorizer()),('tfid', TfidfTransformer())])

transformed_pipe = pipe.fit(support_train_x)

#save pipe for future use
joblib.dump(transformed_pipe, 'pipes/support-tfidf.joblib')

support_count_train = transformed_pipe['count'].transform(support_train_x)

support_tfid_train = transformed_pipe.transform(support_train_x)

support_count_test = transformed_pipe['count'].transform(support_test_x)

support_tfid_test = transformed_pipe.transform(support_test_x)

accuracies(support_tfid_train, support_train_target, support_tfid_test, support_test_target, feature_type='support-tfidf')


# In[25]:



# opposition

transformed_pipe = pipe.fit(oppose_train_x)

joblib.dump(transformed_pipe, 'pipes/oppose-tfidf.joblib')

oppose_count_train = transformed_pipe['count'].transform(oppose_train_x)

oppose_tfid_train = transformed_pipe.transform(oppose_train_x)

oppose_count_test = transformed_pipe['count'].transform(oppose_test_x)

oppose_tfid_test = transformed_pipe.transform(oppose_test_x)

accuracies(oppose_tfid_train, oppose_train_target, oppose_tfid_test, oppose_test_target, feature_type='opposition-tfidf')


# #### DONE
# 
# You can explore the cells below if you want to but it is unrelated to my thesis.
# Head down towards `Embeddings`

# In[34]:


np.savetxt("support_train_target.csv", support_train_target, delimiter=",", fmt='%s')
np.savetxt("support_test_target.csv", support_test_target, delimiter=",", fmt='%s')


# In[150]:


text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(support_train_x, support_train_target)
predicted = text_clf.predict(support_test_x)
print(f"accuracy: {np.mean(predicted == support_test_target)}")

confusion_plot(support_test_target, predicted, classes, labels=labels, normalize=False, title="Linear SVM (1) Confusion Matrix")


# In[144]:


text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LinearSVC( random_state=42, tol=1e-5)),
])

text_clf.fit(support_train_x, support_train_target)
predicted = text_clf.predict(support_test_x)
print(f"accuracy: {np.mean(predicted == support_test_target)}")


confusion_plot(support_test_target, predicted, classes, labels=labels, normalize=False, title="Linear SVM (2) Confusion Matrix")


# In[155]:



text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('logistic',LogisticRegression(solver='liblinear')),
])

text_clf.fit(support_train_x, support_train_target)
predicted = text_clf.predict(support_test_x)
print(np.mean(predicted == support_test_target))
#print(f"probability: {text_clf.predict_proba(support_test_x)}")
confusion_plot(support_test_target, predicted, classes, labels=labels, normalize=False, title="Logistic Regression Confusion Matrix")


# In[6]:



text_rft = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('RFT', RandomForestClassifier(n_estimators = 1500, criterion = "entropy", oob_score = True, max_features= 1)),
])

text_rft.fit(support_train_x, support_train_target)
predicted_rft = text_rft.predict(support_test_x)
print(np.mean(predicted_rft == support_test_target))
#print(f"probability: {text_clf.predict_proba(support_test_x)}")
confusion_plot(support_test_target, predicted_rft, classes, labels=labels, normalize=False, title="RandomForestClassifier Confusion Matrix")



# In[9]:


with open('model.pkl', 'wb') as f:  # open a text file
    dump(text_rft, f) 

with open('model.pkl', 'rb') as f:
    new_clf = load(f)


# In[10]:


for i in zip(text_rft.predict_proba(support_test_x) , text_rft.predict(support_test_x)):
    print(i)


# In[154]:


text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('KNN', KNeighborsClassifier()),
])

text_clf.fit(support_train_x, support_train_target)
predicted = text_clf.predict(support_test_x)
print(f" accuracy {np.mean(predicted == support_test_target)}")
confusion_plot(support_test_target, predicted, classes, labels=labels, normalize=False, title="KNearest Neighbours Confusion Matrix")


# ### Embeddings
# 
# This is th Embeddings part of the experiment. Here I train models with embeddings.
# I use Hugging faces sentence-transformers embeddinggs

# In[27]:


sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
testset['embeddings'] = ""
testset['embeddings'] = testset['prompt'].map(sentence_model.encode)
testset['embeddings']

support_train_x = np.array(testset["embeddings"].loc[(testset["brief_type"]=="support") & (testset["data_type"]=="train") ].to_list())  #  np.array(testset['file_path'].to_list())
opposition_train_x = np.array(testset["embeddings"].loc[(testset["brief_type"]=="opposition") & (testset["data_type"]=="train") ].to_list())  #  np.array(testset['file_path'].to_list())

support_test_x = np.array(testset["embeddings"].loc[(testset["brief_type"]=="support") & (testset["data_type"]=="test") ].to_list())  #  np.array(testset['file_path'].to_list())
opposition_test_x = np.array(testset["embeddings"].loc[(testset["brief_type"]=="opposition") & (testset["data_type"]=="test") ].to_list()) 


support_train_target = np.array( testset["completion"].loc[(testset["brief_type"]=="support") & (testset["data_type"]=="train") ].to_list())  # np.array(testset['label'].to_list())
opposition_train_target=  np.array(testset["completion"].loc[(testset["brief_type"]=="opposition") & (testset["data_type"]=="train") ].to_list())  # np.array(testset['label'].to_list())


support_test_target = np.array( testset["completion"].loc[(testset["brief_type"]=="support") & (testset["data_type"]=="test") ].to_list())  # np.array(testset['label'].to_list())
opposition_test_target =  np.array(testset["completion"].loc[(testset["brief_type"]=="opposition") & (testset["data_type"]=="test") ].to_list()) 


# In[28]:


accuracies(support_train_x, support_train_target, support_test_x, support_test_target,feature_type='support-sentence_embeddings')


# In[29]:


accuracies(opposition_train_x, opposition_train_target, opposition_test_x, opposition_test_target,feature_type='opposition-sentence_embeddings')


# ### Combination Deep Sets

# https://github.com/dpernes/deepsets-digitsum
# 
# https://github.com/manzilzaheer/DeepSets/blob/master/DigitSum/image_sum.ipynb
# 
# https://paperswithcode.com/paper/deep-sets
# 
# https://github.com/lucidrains/perceiver-pytorch
# 
# https://www.youtube.com/watch?v=P_xeshTnPZg
# 
# https://paperswithcode.com/method/set-transformer
# 
# https://arxiv.org/pdf/1810.00825.pdf
# 
# https://arxiv.org/abs/1910.02421
# 
# https://arxiv.org/pdf/1703.06114.pdf
# 
# https://huggingface.co/docs/transformers/model_doc/perceiver
# 
# https://www.youtube.com/watch?v=Xe7VT8-kDzg
# 
# https://github.com/krasserm/perceiver-io
# 
# https://www.youtube.com/watch?v=YBkOILybiNo
# 
# https://www.youtube.com/watch?v=9ymIqU4XnhY
# 
# https://paperswithcode.com/paper/set-transformer-a-framework-for-attention#code
# 
# https://www.inference.vc/deepsets-modeling-permutation-invariance/
# 
# https://medium.com/@albertoarrigoni/paper-review-code-deep-sets-5f87d335f16f
# 
# https://medium.com/@albertoarrigoni/paper-review-code-set-transformer-b9750e5c3fdb
# 
# https://www.youtube.com/watch?v=wTZ3o36lXoQ&t=199s
