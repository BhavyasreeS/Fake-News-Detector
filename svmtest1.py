#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fake news detection
The SVM model
"""

from getEmbeddingstest import getEmbeddings
import numpy as np

import matplotlib.pyplot as plt
import scikitplot.plotters as skplt
import os
from sklearn.externals import joblib



def plot_cmat(yte, ypred):
    '''Plotting confusion matrix'''
    skplt.plot_confusion_matrix(yte,ypred)
    plt.show()

# Read the data
if not os.path.isfile('./xtr.npy') or \
    not os.path.isfile('./xte1.npy') or \
    not os.path.isfile('./ytr.npy') or \
    not os.path.isfile('./yte1.npy'):
    xtr,xte,ytr,yte = getEmbeddings("datasets/train2.csv")
    np.save('./xtr', xtr)
    np.save('./xte1', xte)
    np.save('./ytr', ytr)
    np.save('./yte1', yte)

#xtr = np.load('./xtr.npy')
xte = np.load('./xte1.npy')
#ytr = np.load('./ytr.npy')
yte = np.load('./yte1.npy')

filename='savedsvm.sav'
loaded_model = joblib.load(filename)

y_pred = loaded_model.predict(xte)
m = yte.shape[0]
n = (yte != y_pred).sum()
print("Accuracy = " + format((m-n)/m*100, '.2f') + "%")   # 88.42%

# Draw the confusion matrix
plot_cmat(yte, y_pred)
print("y pred is")
print( y_pred)
