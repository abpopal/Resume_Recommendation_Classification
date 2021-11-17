# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 17:10:29 2021

@author: Abdul Baqi Popal
"""


# we will use KNN Algo to find and classify docs according to their similarity
import docx2txt
import pandas as pd 
import re
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

#import data set

dataset = pd.read_csv('Resume.csv')




def cleanResume(resumeText):
    resumeText = resumeText.lower()
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]',r' ', resumeText) 
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText

dataset['cleaned_resume'] = dataset.Resume_str.apply(lambda x: cleanResume(x))

from sklearn.preprocessing import LabelEncoder


var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    dataset[i] = le.fit_transform(dataset[i])
    
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


requiredText = dataset['cleaned_resume'].values
requiredTarget = dataset['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
prediction = le.inverse_transform(prediction) # this will convert the encoder back to its original label 

resume = docx2txt.process('resume.docx')
classify = clf.predict(word_vectorizer.transform([resume]))
classify = le.inverse_transform(classify)

print("Resume belongs to: ", classify[0])

