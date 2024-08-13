
# coding: utf-8

# ### Note 
# You will have to run this notebook twice. One for Support and the Other for Opposition
# 

# In[1]:


from datasets import Dataset , load_dataset, DatasetDict
import pandas as pd
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding , AutoTokenizer
from transformers import AutoModelForSequenceClassification , TrainingArguments , AutoConfig
from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn.functional as f
import evaluate
import numpy as np
import os
from torch.utils.data import DataLoader
import evaluate
import evaluate
import time
import wandb
import random 
from transformers import TrainingArguments, Trainer , AutoModelForSequenceClassification

from datasets import Features , ClassLabel, Value, Sequence

roberta_checkpoint = "roberta-large"

mistral_checkpoint = "mistralai/Mistral-7B-v0.1"
bert_checkpoint = "bert-base-uncased"

llama_checkpoint = "meta-llama/Llama-2-7b-hf"
MAX_LEN = 512 
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
torch.distributed.is_available()


# ### Dataset and Functions
# 
# In this tab retrive the dataset relevant for the experiment.
# You only have to run the cell below once.
# It is important know that the cell below produces two types of dataframes, "support" and "opposition".
# 
# 
# This tab also contains most of the functinos used in this Notebook

# In[2]:


def decision2label(decision):
    if  "grant" in decision:
        return 1
    elif "deny" in decision:
        return 0
    else:
        print(f"error occured with decision: {decision} ",)
        exit("Invalid decision")


def test_metrics(model, dataloader):
    acc = evaluate.load("accuracy")
    preci = evaluate.load("precision")
    recall = evaluate.load("recall")

    csv = {'brief':[],'predict':[], 'score':[], 'truth':[]}

    model.eval()
    for batch in dataloader:
        briefs = batch['file_name']
        inputs = {k: v.to(device) for k, v in batch.items() if k != "file_name"}
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        propabilities = f.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
        acc.add_batch(predictions=predictions, references=inputs["labels"])
        preci.add_batch(predictions=predictions, references=inputs["labels"])
        recall.add_batch(predictions=predictions, references=inputs["labels"])

        csv['brief'].extend(briefs)
        labels = lambda x: "grant" if x == 1 else "deny"
        predict = list(map(labels, predictions))
        csv['predict'].extend(predict) 
        csv['score'].extend(propabilities[:,1].cpu().numpy())
        csv['truth'].extend(list(map(labels, inputs["labels"].cpu().numpy())))

    return {'accuracy': acc.compute()['accuracy'],
            'precision': preci.compute()['precision'], 
            'recall': recall.compute()['recall'],
            'csv': csv}

#TESTSET = "../dataset/testset.csv"

UNPAIRED_PATH = '../dataset/testset.csv'

testset = pd.read_csv(UNPAIRED_PATH, index_col=0)

testset = testset.loc[testset['data_type'] == 'train']

# randomly set 20% of the data to test
testset['data_type'] = testset['data_type'].apply(lambda x: "test" if random.random() < 0.2 else "train")

testset['labels'] = testset['completion'].apply(decision2label)

train = testset.loc[testset['data_type'] == 'train']
test = testset.loc[testset['data_type'] == 'test']

support_train = train.loc[train['brief_type'] == "support"]
support_test = test.loc[test['brief_type'] == "support"]

oppo_train = train.loc[train['brief_type'] == "opposition"]
oppo_test = test.loc[test['brief_type'] == "opposition"]

testset

# try putting support and train together



# ### Define tokenizers and Dataloaders
# 
# You will have to edit and run the below cells several times for each datatype and model.
# 
# As of the time this note was taken, the below cells uses bert and train opposition.
# 
# If you want to use mistral or use train on support or increase/decrease the context size then you will have to various variable names.
# 
# For exmaple:
# ```model_type = "bert"```. This sets the model type to bert. You can change this to mistral
# `key = "opposition"`. This sets the input type to oppositon. You can change this to support.
# 
# 

# In[3]:



model_type = "bert"

key = "opposition"

# Remember to change this

features = Features({ 'prompt' : Value(dtype='string'),
                     'completion': ClassLabel(num_classes=3, names=['deny', 'grant', 'TBD'],  id=None),
                     'brief_type' : ClassLabel(num_classes=2, names=["support", "opposition"], id=None),
                        'data_type' : ClassLabel(num_classes=2, names=["train", "test"], id=None),
                        'file_path' : Value(dtype='int64') ,
                        'file_name' : Value(dtype='string'),   
                        'labels' : Value(dtype='int64')
                     })

# can change the argument
if key == "support":    
    dataset_train = Dataset.from_pandas(support_train, preserve_index=False , features= features )
    dataset_test = Dataset.from_pandas(support_test, preserve_index=False,  features= features)
else: #  key == opposition
   
    dataset_train = Dataset.from_pandas(oppo_train, preserve_index=False, features= features)
    dataset_test = Dataset.from_pandas(oppo_test, preserve_index=False, features= features)

lr = 1e-5

dataset = DatasetDict()


dataset['train'] = dataset_train
dataset['test'] = dataset_test


if model_type == "mistral":

    tokenizer = AutoTokenizer.from_pretrained(mistral_checkpoint, add_prefix_space=True, device=device)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token


    config = AutoConfig.from_pretrained(mistral_checkpoint)
    max_input_size =  1024

    def tokenize_function(examples):
        return tokenizer(examples['prompt'], truncation= True, padding="max_length" , max_length=max_input_size)

    #mistral_data_collator = DataCollatorWithPadding(tokenizer)

else:
    tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)

    def tokenize_function(briefs):
     return tokenizer(briefs["prompt"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(200))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(200))

tokenized_datasets = tokenized_datasets.remove_columns(["completion","prompt","brief_type","data_type", "file_path", ]) # "file_name"])
tokenized_datasets.set_format("torch")


train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=16, )
eval_dataloader = DataLoader(tokenized_datasets["test"], batch_size=16, ) 


print(tokenized_datasets)
testset


# ## Model
# 
# There are two models available to be used in this experiment.
# Bert and Mistral.
# 
# Only run the cell that corresponds to the model you want to use.
# 
# In the cells below, you can also configure the model and it's hyperparameters as you see fit.
# 
# Important note:
# Mistral is quite large and could not fit in the current GPU capacity for this experiment.

# Bert

# In[4]:


lr = 1e-5
model = AutoModelForSequenceClassification.from_pretrained(bert_checkpoint, num_labels=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
num_params = model.num_parameters()
print(f"The model has {num_params} parameters.")


# Mistral (Do not run this)

# In[4]:



# model =  AutoModelForSequenceClassification.from_pretrained(
#   pretrained_model_name_or_path=mistral_checkpoint,
#   num_labels=2,
#   #use_flash_attention_2=True,
#   torch_dtype= torch.bfloat16,
# #  device_map="auto"
# ).to(device)

# model.config.pad_token_id = model.config.eos_token_id

# mistral_peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS, r=2, lora_alpha=16, lora_dropout=0.1, bias="none", 
#     target_modules=[
#         "q_proj",
#         "v_proj",
#     ],
# )

# model = get_peft_model(model, mistral_peft_config)
# model.print_trainable_parameters()

# lr = 1e-5
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)



# ## Train and Test
# 
# This project is connected to Weight and Biases [(wandb)](https://wandb.ai/site)
# 
# So make sure to pip install and sign up to wand if you want to see the training graphs.
# 
# In the cell below you can change the num_epochs.
# 
# 

# In[5]:



# what does get_scheduler do?

from transformers import get_scheduler

num_epochs = 50
num_training_steps = num_epochs * len(train_dataloader)

name = "Support" if key == "support" else "Opposition"

wandb.init(
    # set the wandb project where this run will be logged
    project="LLM_TOTURIAL",  
    name= f"{name}-{bert_checkpoint}",#f"Opposition-mistral-7B-v0.1-1-Tokensize:{max_input_size}",
    # track hyperparameters and run metadata
    config={
    "optimizer": "AdamW",
    "lr": lr,

    "dataset": "single-supports",
    "epochs": num_epochs,
    }
)



# ### Training and Evaluation
# 
# Simply put this is where training and evaluation takes place.
# At every epoch i send the accumulated average loss, metrics like precision, recall and accuracy to wandb
# 
# I save the model to disk that produces the highest accuracy, aswell as the prediction files.

# In[6]:


from tqdm.auto import tqdm
progress_bar = tqdm(range(num_training_steps))

# i wonder if the outputs.loss is the same as loss_fn(outputs, labels)
# Try to log the values 

best_valid_acc = 0.0
model.train()
print("Training model")
for epoch in range(num_epochs):
    acc = evaluate.load("accuracy")
    average_loss = 0
    for batch in train_dataloader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "file_name"}
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        
        average_loss += loss.item()
        

        optimizer.step()
       
        optimizer.zero_grad()
        progress_bar.update(1)

        # get the predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        acc.add_batch(predictions=predictions, references=inputs["labels"])

    accuracy_per_epoch = acc.compute()
    print(f"Epoch {epoch} completed")
    print(f"Accuracy: {accuracy_per_epoch}")
    avg_loss = average_loss / len(train_dataloader)
    print(f"loss : {avg_loss}")

    print("Evaluating model on test set")
    metrics = test_metrics(model, eval_dataloader)
    csv = metrics["csv"]
    csv = pd.DataFrame(csv)
    print(metrics)
    
    wandb.log({"loss_per_epoch": avg_loss , 
               "accuracy_per_epoch": accuracy_per_epoch,
               "test_accuracy" :metrics["accuracy"],
                "test_recall": metrics["recall"],
                "test_precision": metrics["precision"],
               })
    
    if metrics["accuracy"] > best_valid_acc:
        best_valid_acc = metrics["accuracy"]
        print("Saving model")
        model.save_pretrained(f"../models/LLM-{model_type}-{key}-test")
        csv.to_csv(f"../predictions/LLM-{model_type}-{key}-test.csv", index=False)
    
wandb.finish()



