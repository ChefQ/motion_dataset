# %%
import pandas as pd
import re
import os
from zipfile import ZipFile 
import numpy as np
import ast
import argparse
from functools import reduce

# %% [markdown]
# 1. Knn make predicted denies negative
# 2. transfomr the confidence scores to probabillities for denies and then convert to log odds
# 3. for probabilites convert to log odds

# %%


PAIRED_PATH = './dataset/paired_testset.csv'
paired = pd.read_csv(PAIRED_PATH, index_col=0)
paired['support_file'] = paired['support_file'].apply(ast.literal_eval)
paired['opposition_file'] = paired['opposition_file'].apply(ast.literal_eval)

def get_folderid(brief_type, query, paired):
    for index, row in paired.iterrows():
        briefs = row[brief_type]
        if query in briefs:
            return row['folder_id']      
    return None


# for the KNN make the 1.0 to 0.9 and 0.0 to 0.1

def add_folderId(path, includeLogOdds=False, changeDenyProb = False, alterProb=False):

    # assert supp or oppo in path both not both
    if not  ( ('supp' in path) ^ ('oppo' in path )):
        raise ValueError('path must contain either supp xor oppo')
    
    file_type = 'support_file' if 'supp' in path else 'opposition_file'
    

    df = pd.read_csv(path)
    
    df['newScore'] = df['score']

    log_odds = lambda prob: np.log((prob + 1e-6 )/(1-prob + 1e-6))

    newScore = lambda row: 1 - row['newScore']  if 'deny' in row['predict'] else row['newScore']

    if alterProb:
        df['newScore'] = df['newScore'].apply(lambda x: 0.9 if x == 1.0 else 0.1 if x == 0.0 else x)
        
    if changeDenyProb:
        df['newScore'] = df.apply(newScore, axis=1)

    if includeLogOdds:
        
        df['log_odds'] = df['newScore'].apply(log_odds)

    else:

        df['log_odds'] = df['newScore']


    df['folder_id'] = df['brief'].apply(lambda x: get_folderid(file_type,x, paired))
    return df


def derive_fuse(df):

    grouped = df.groupby('folder_id')
    grouped = grouped.agg({'log_odds': 'sum', 'newScore': 'mean', 'folder_id': 'first', 'truth': 'first'})

    predict = lambda x: 'deny' if x < 0 else 'grant'
    grouped['predict'] = grouped['log_odds'].apply(predict)
    return grouped

def derive_first(df):
    
    predict = lambda x: 'deny' if x < 0 else 'grant'
    
    grouped = df.groupby('folder_id')

    grouped = grouped.agg({'brief': first , })


    new_df = df[df['brief'].isin(grouped['brief'])].copy()


    new_df['predict'] = new_df['log_odds'].apply(predict)


    return new_df


def first(series):
    def get_num(val):
        return int(re.match(r".+\.(\d+)\.\d+\.txt", val).group(1))
    return reduce(lambda x, y: x if get_num(x) < get_num(y) else y, series)

    


# %%


parser = argparse.ArgumentParser(description='Combination methods for predictions: fuse and first ')
parser.add_argument('-type', help=f'Types are fuse and first')
parser.add_argument('-path', help='Path to the predictions')
parser.add_argument('-all', help='predict all predictions in the folder', action='store_true')
parser.add_argument('-alterProb', help='Change the probabilities to 0.9 and 0.1', action='store_true')
parser.add_argument('-includeLogOdds', help='Include log odds in the predictions', action='store_true')
parser.add_argument('-changeDenyProb', help='Change the deny probabilities to 1 - prob', action='store_true')

arg = parser.parse_args()


if arg.all == True:



    pairs = {
    ('./predictions/SGD-sentence_embeddings-supppredictions.csv', './predictions/SGD-sentence_embeddings-oppopredictions.csv'): {},
    ('./predictions/LinearSVC-sentence_embeddings-supppredictions.csv', './predictions/LinearSVC-sentence_embeddings-oppopredictions.csv'): {}, 
    ('./predictions/RFT-sentence_embeddings-supppredictions.csv', './predictions/RFT-tfidf-oppopredictions.csv'): {'includeLogOdds': True, 'changeDenyProb': True}, 
    ('./predictions/RFT-sentence_embeddings-oppopredictions.csv', './predictions/RFT-tfidf-oppopredictions.csv'): {'includeLogOdds': True, 'changeDenyProb': True}, 
    ('./predictions/Logistic-tfidf-supppredictions.csv', './predictions/Logistic-tfidf-oppopredictions.csv'): {'includeLogOdds': True, 'changeDenyProb': True}, 
    ('./predictions/SGD-tfidf-supppredictions.csv', './predictions/SGD-tfidf-oppopredictions.csv'): {}, 
    ('./predictions/RFT-tfidf-oppopredictions.csv', './predictions/RFT-tfidf-supppredictions.csv'): {'includeLogOdds': True, 'changeDenyProb': True}, 
    ('./predictions/LinearSVC-tfidf-oppopredictions.csv', './predictions/LinearSVC-tfidf-supppredictions.csv'): {}, 
    ('./predictions/KNN-tfidf-oppopredictions.csv', './predictions/KNN-tfidf-supppredictions.csv'): {'includeLogOdds': True, 'changeDenyProb': True}, 
    ('./predictions/Logistic-sentence_embeddings-supppredictions.csv', './predictions/Logistic-sentence_embeddings-oppopredictions.csv'): {'includeLogOdds': True, 'changeDenyProb': True},
    ('./predictions/KNN-sentence_embeddings-supppredictions.csv', './predictions/KNN-sentence_embeddings-oppopredictions.csv'): {'includeLogOdds': True, 'changeDenyProb': True},
    ('./predictions/Logistic-tfidf-oppopredictions.csv', './predictions/Logistic-tfidf-supppredictions.csv'): {'includeLogOdds': True, 'changeDenyProb': True},
    ('./predictions/LLM-bert-base-uncased-supppredictions.csv', './predictions/LLM-bert-base-uncased-oppopredictions.csv'): {'includeLogOdds': True, 'changeDenyProb': True}
    }

    if arg.type == 'fuse':
        if not os.path.exists('./predictions/fuse'):
            os.makedirs('./predictions/fuse')

        for (path1, path2), args in pairs.items():
            print(path1, path2)

            df1 = add_folderId(path1, **args)
            grouped1 = derive_fuse(df1)
            grouped1.to_csv(path1.replace('/predictions' , '/predictions/fuse').replace('.csv', '_fuse.csv'))

            df2 = add_folderId(path2, **args)
            grouped2 = derive_fuse(df2)
            grouped2.to_csv(path2.replace('/predictions' , '/predictions/fuse').replace('.csv', '_fuse.csv'))

            df = pd.concat([df1, df2])
            grouped = derive_fuse(df)

            file_type = 'supp' if 'supp' in path1 else 'oppo'
            grouped.to_csv(path1.replace('/predictions' , '/predictions/fuse').replace(file_type, "both").replace('.csv', '_fuse.csv'))
        
        print('done')
    
    elif arg.type == 'first':
        
        if not os.path.exists('./predictions/first'):
            os.makedirs('./predictions/first')

        for (path1, path2), args in pairs.items():
            print(path1, path2)

            df1 = add_folderId(path1, **args)
            new_df1 = derive_first(df1)
            new_df1.to_csv(path1.replace('/predictions' , '/predictions/first').replace('.csv', '_first.csv'))

            df2 = add_folderId(path2, **args)
            new_df2 = derive_first(df2)
            new_df2.to_csv(path2.replace('/predictions' , '/predictions/first').replace('.csv', '_first.csv'))

        print('done')

    else:
        raise ValueError('Type must be fuse or first')
    
else:
    
    if arg.type == 'fuse':
        if not os.path.exists('./predictions/fuse'):
            os.makedirs('./predictions/fuse')

        args = {}
        if arg.includeLogOdds:
            args['includeLogOdds'] = True
        if arg.changeDenyProb:
            args['changeDenyProb'] = True
        if arg.alterProb:
            args['alterProb'] = True
            
        df = add_folderId(arg.path)
        grouped = derive_fuse(df)
        grouped.to_csv(arg.path.replace('/predictions' , '/predictions/fuse').replace('.csv', '_fuse.csv'))
        print('done')
        
    elif arg.type == 'first':
        if not os.path.exists('./predictions/first'):
            os.makedirs('./predictions/first')

        df = add_folderId(arg.path)
        new_df = derive_first(df)
        new_df.to_csv(arg.path.replace('/predictions' , '/predictions/first').replace('.csv', '_first.csv'))
        print('done')

    else:
        raise ValueError('Type must be fuse or first')


