from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import platform
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import pickle

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

file_path = os.path.join("/Users/xiezhijun/Desktop/Gaodun/Machine Learning/ML JOB/BERT/BERT-Classification-Tutorial/data", 'train_taxdata_0107_stopwords.csv')
with open(file_path, 'r') as f:
    reader = f.readlines()
examples = []
labels = []
for index, line in enumerate(reader):
    guid = 'train-%d'%index
    split_line = line.strip().split(',')
    text_a = tokenization.convert_to_unicode(split_line[0])
    label = tokenization.convert_to_unicode(split_line[1])

    # label = split_line[1]
    labels.append(label)
    examples.append(InputExample(guid=guid, text_a=text_a,
                                 text_b=None, label=label))

print("text_a:", text_a)
print("labels:", labels)