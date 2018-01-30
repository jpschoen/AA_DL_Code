#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script evaluates the use of word2vec and GloVe embeddings for text 
classification of escort data

Parts of this code are borrowed from Nadbor Drozd
http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/
"""
#loading modules
import os
from tabulate import tabulate
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from keras.utils import to_categorical
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedShuffleSplit

# loading data sets. ht_vectors.txt is an output of glove trained on HT data
TRAIN_SET_PATH = "data_clean/top_5percent.csv"
GLOVE_PATH = "glove_pretrained/ht_vectors.txt" #(100d)



# Escort data
X, y = [], []
ht_df = pd.read_csv(TRAIN_SET_PATH)

labels_index = {}  # dictionary mapping label name to numeric id
labels = ht_df['Phone_Parsed']  # list of label ids
for string in labels:
    labels_index.setdefault(string, len(labels_index))
labels =  np.asarray([labels_index[Phone_Parsed] for Phone_Parsed in labels])
X, y = np.array(ht_df['Post']), labels
print("total examples %s" % len(y))

#Glove
with open(GLOVE_PATH, "rb") as lines:
    word2vec = {line.split()[0]: np.array(map(float, line.split()[1:]))
               for line in lines}
               
glove = {}
all_words = set(w for words in X for w in words)
with open(GLOVE_PATH, "rb") as infile:
    for line in infile:
        parts = line.split()
        word = parts[0]
        nums = map(float, parts[1:])
        if word in all_words:
            glove[word] = np.array(nums)
            
            
# train Word2vec
model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
       
#Define text classification models: Multinomial Naive Bayes and SVM        
# with either pure counts or tfidf features

mult_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
mult_nb_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])

svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])


# The next part of code builds features by averaging word vectors for all words in a text
# It builds a sklearn-compatible transformer that is initialised with a word -> vector dictionary.

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = len(word2vec.itervalues().next())
    
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])
    
# and a tf-idf version of the same

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

                
# We stack embeddings in extra tree classifier
etree_glove = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
#etree_glove_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove)), 
#                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])

etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)), 
                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])
#etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)), 
#                        ("extra trees", ExtraTreesClassifier(n_estimators=200))])

#I added extra tree classifier without word embeddings
etree = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("extra trees", ExtraTreesClassifier(n_estimators=200))])
etree_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("extra trees", ExtraTreesClassifier(n_estimators=200))])


all_models = [
    ("mult_nb", mult_nb),
    ("mult_nb_tfidf", mult_nb_tfidf),
    ("svc", svc),
    ("svc_tfidf", svc_tfidf),
    ("etree_glove", etree_glove), 
#    ("etree_glove_tfidf", etree_glove_tfidf),
    ("etree_w2v", etree_w2v),
#    ("etree_w2v_tfidf", etree_w2v_tfidf),
    ("etree", etree),
    ("etree_tfidf", etree_tfidf),
]

# Calculating accuracy for each model
scores = sorted([(name, cross_val_score(model, X, y, cv=5).mean()) 
                 for name, model in all_models], key=lambda (_, x): -x)
print tabulate(scores, floatfmt=".4f", headers=("model", 'score'))


fig = plt.figure(figsize=(15, 6))
sns.barplot(x=[name for name, _ in scores], y=[score for _, score in scores])
fig.suptitle('Text classification models')
plt.xlabel('Models')
plt.ylabel('Accuracy')
fig.savefig('Comparison_HT.png')
plt.show()
