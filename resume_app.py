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


processed_cv = preprocess(docx2txt.process('resume.docx'))

unProcessed_cv = docx2txt.process('resume.docx')

processed_jobD = preprocess(docx2txt.process('jobdescription.docx'))

unProcessed_jobD = docx2txt.process('jobdescription.docx')

unprocessed_docs = [unProcessed_cv, unProcessed_jobD]

processed_docs = [processed_cv, processed_jobD]


cv_processed = CountVectorizer()
cv_Unprocessed = CountVectorizer()



count_matrix_processed = cv_processed.fit_transform(processed_docs)
count_matrix_unprocessed = cv_Unprocessed.fit_transform(unprocessed_docs)

similarity_processed = cosine_similarity(count_matrix_processed)
similarity_unprocessed = cosine_similarity(count_matrix_unprocessed)


process_percentage = round(similarity_processed[0][1] * 100, 2)
unprocess_percentage = round(similarity_unprocessed[0][1] * 100, 2)

print("processed docs similarity: ", process_percentage)
print("unprocess docs similarity: ", unprocess_percentage)
"""

resume = docx2txt.process('resume.docx')

job_description = docx2txt.process('jobdescription.docx')


text = [resume, job_description]

cv = CountVectorizer()

count_matrix = cv.fit_transform(text)

similarity = cosine_similarity(count_matrix)

match_percentage = round(similarity[0][1] * 100, 2)

print(match_percentage)

"""