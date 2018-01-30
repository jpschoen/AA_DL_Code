# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 16:03:42 2017

@author: johnpschoeneman
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


#read in  Data and list features
ht_df = pd.read_csv("top_75percent.csv")



print(list(ht_df))

posts = ht_df['Post']
digits = ht_df['Phone_Parsed']

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(posts)
print(vectors.shape)


X_train, X_test, y_train, y_test = train_test_split(vectors, digits, test_size=0.2, random_state=19)


print(X_train.shape, X_test.shape)


svm = LinearSVC()
svm.fit(X_train, y_train)
 

predictions = svm.predict(X_test)
print(list(predictions[0:10]))

print(y_test[:10])


print(accuracy_score(y_test, predictions))


