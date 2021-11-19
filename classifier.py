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
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import figure
#import data set

dataset = pd.read_csv('Resume.csv')


# some data visualization 
import seaborn as sns

sns.countplot(y="Category", data=dataset)

count = dataset['Category'].value_counts()

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
#accuracy
accuracy = accuracy_score(y_test, prediction, normalize=False)
#some confusin matrix visualization

cm = confusion_matrix(prediction, y_test)


matrix = plot_confusion_matrix(clf, X_test, y_test, cmap=plt.cm.Reds )
matrix.ax_.set_title("confusion matrix", color='black')
plt.xlabel('predicted Label', color='black')
plt.ylabel('True Label', color='black')
plt.gcf().axes[0].tick_params(color='black')
plt.gcf().axes[1].tick_params(color='black')
plt.gcf().set_size_inches(10,6)
plt.show()

#display the output

prediction = le.inverse_transform(prediction) # this will convert the encoder back to its original label 

resume = docx2txt.process('resume.docx')
classify = clf.predict(word_vectorizer.transform([resume]))
classify = le.inverse_transform(classify)

print("Resume belongs to: ", classify[0])


