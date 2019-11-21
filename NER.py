# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 20:25:44 2019

@author: Ahmed Hatem
"""
#%%
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
#%%
dataset_path = r"C:/Users/Ahmed Hatem/Desktop/Ahmed/Nile university/Courses/NLP/Project/Data/NER Annotated Corpus/ner_dataset.csv"
embeddings_path = r"C:/Users/Ahmed Hatem/Desktop/Ahmed/Nile university/Courses/Practical Data Mining/Assignments/Assi 3/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"
#%%
df = pd.read_csv(dataset_path, encoding = "ISO-8859-1", error_bad_lines=False)

embeddings_index = {}
wv_from_bin = KeyedVectors.load_word2vec_format(embeddings_path, binary=True) 
for word, vector in zip(wv_from_bin.vocab, wv_from_bin.vectors):
    coefs = np.asarray(vector, dtype='float32')
    embeddings_index[word] = coefs
    
#%% Preprocess dataset
# Lower case all words
df["Word"] = df["Word"].apply(str.lower)

#%%
def get_wordvec(word):
    try:
        return embeddings_index[word]
    except KeyError:
        return np.zeros([300,])

#%% Construct sentences as embedded vectors, List of lists of tuples
corpus_list = []
for indx, row in df.iterrows():
    if "sentence" in str(row["Sentence #"]).lower():
        corpus_list.append([])
    corpus_list[-1].append((row["Word"], get_wordvec(row["Word"]), row["Tag"]))
    


