# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:44:54 2021

@author: Abdul Baqi Popal
"""


import docx2txt
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
import string
import re
from os import walk
import shutil, os


# -------------------------------data preprocess 
stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

# preprocessor method 

def preprocess(text):
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    x = text.split()
    new_text = []
    for word in x:
        if word not in stop_words:
            new_text.append(stemmer.stem(word))
            
    text = ' '.join(new_text)
    return text

# recommender - returns percentage of similarity between cv and Job Description
def recommender(cv, jobD):
        
        docs = [cv, jobD]
        count_vector = CountVectorizer()
        count_matrix = count_vector.fit_transform(docs)
        
        similarity = cosine_similarity(count_matrix)
        
        return round(similarity[0][1] * 100, 2)

        

cv_process = preprocess(docx2txt.process("Elliot-Alderson-Resume-Software-Developer-2.docx"))
cv_unprocess = docx2txt.process("Elliot-Alderson-Resume-Software-Developer-2.docx")

jobD_process = preprocess(docx2txt.process("Software Engineer Job Description.docx"))
jobD_unprocess = docx2txt.process("Software Engineer Job Description.docx")

print("processed data percentage",recommender(cv_process, jobD_process))
print("Unprocessed data percentage",recommender(cv_unprocess, jobD_unprocess))

f = []
recommended = []
jobDescription = preprocess(docx2txt.process("Software Engineer Job Description.docx"))
for (dirpath, dirnames, filenames) in walk('received/'):
    f.extend(filenames)
    break
#print("done ", f)
    
print("\nresumes received")
print("\nrecommender running")  
for x in f:
    filename = 'received/'+x
    #print("file name is ", filename)
    #print("the value for x is: ", x)
    match = recommender(preprocess(docx2txt.process(filename)),jobDescription)
    print("\nmatch for: ", x, "is: ", match)
    if  match > 45:
        recommended.append(filename)
        
   
#print(recommended)
for f in recommended:
    shutil.copy(f, 'recommended')
    
print("\nresumes saved to recommended folder!!")
