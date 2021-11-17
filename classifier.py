# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 17:10:29 2021

@author: Abdul Baqi Popal
"""


# we will use KNN Algo to find and classify docs according to their similarity
import docx2txt
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
import string
import re

#import data set

dataset = pd.read_csv('Resume.csv')

dataset['cleaned_resumes'] = ''
