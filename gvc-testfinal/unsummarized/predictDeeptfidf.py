
# coding: utf-8

# In[1]:



# make shell call on the command line
import subprocess

# make a call to the command line

#models = ["KNN","LinearSVC", "Logistic", "RFT", "SGD"]
#feature_types = ['tfidf', 'sentence_embeddings']
#
#for model in models:
    #for feature_type in feature_types:
        #subprocess.call(f"python3 p-script.py --data test_data/testset.csv --model_name {model} --feature {feature_type}", shell=True)
#
#
#subprocess.call("python3 p-script.py --data test_data/testset.csv --model_name LLM", shell=True)
#subprocess.call("python3 p-script.py --data test_data/paired_testset.csv --combine --feature sentence_embeddings", shell=True)
subprocess.call("python3 p-script-tfidf.py --data test_data/paired_testset.csv --combine --feature tfidf", shell=True)


# In[5]:




