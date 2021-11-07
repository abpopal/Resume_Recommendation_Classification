# -*- coding: utf-8 -*-
"""
Created on Sun Nov  7 12:44:54 2021

@author: Abdul Baqi Popal
"""


import docx2txt
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


resume = docx2txt.process('resume.docx')

job_description = docx2txt.process('jobdescription.docx')


text = [resume, job_description]

cv = CountVectorizer()

count_matrix = cv.fit_transform(text)

similarity = cosine_similarity(count_matrix)

match_percentage = round(similarity[0][1] * 100, 2)

print(match_percentage)