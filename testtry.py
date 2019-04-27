# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 09:41:41 2019

@author: sbhav
"""
import re
import string
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from nltk.corpus import stopwords
import pandas as pd
from sklearn.externals import joblib

def textClean(text):
    """
    Get rid of the non-letter and non-number characters
    """
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", str(text))
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return (text)
def cleanup(text):
    text = textClean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text
def constructLabeledSentences(data):
    sentences = []
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % index]))
    return sentences
path = 'datasets/trains.csv'
data = pd.read_csv(path)
data.loc[:, 'text'] = cleanup(data.loc[:,'text'])
model= Doc2Vec.load("doc2vec.model")
v1 = model.infer_vector(data['text'])
filename='savedsvms.sav'
loaded_model = joblib.load(filename)
v1=v1.reshape(1,-1)
y_pred = loaded_model.predict(v1)
print(y_pred)
