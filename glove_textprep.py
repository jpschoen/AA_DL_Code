#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 10:59:53 2017

@author: johnpschoeneman
"""
import os
import pandas as pd

#read in data
ht_df = pd.read_csv("data_clean/clean_1to8.csv", encoding = 'utf=8')
#concatenate and write posts to text file.
file_ht = open("ht_text.txt", "wb")
text = ht_df['Post']
text = text.str.cat(sep=' ')
file_ht.write(text.encode('utf-8'))
file_ht.close