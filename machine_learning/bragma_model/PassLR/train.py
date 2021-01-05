# -*- coding: utf-8 -*-
# 标签是用is_hit_risklist

from sklearn import tree
import pydotplus
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn import preprocessing
import pickle
import json
import os

data1 = pd.read_excel("query_result-79.xlsx")
data = pd.read_table("test_model_data.txt", sep="\t")

new_list = []
for element in data1.columns:
    new_list.append(element.split(".")[1])
data.columns = new_list

del data["apply_id"]
del data["apply_time"]
del data["channel"]
del data["user_id"]
del data["id_card_md5"]
del data["dt"]
data = data.fillna(0)
X = data.iloc[:, 1:]

X_columns = X.columns

min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)
Y = data[["is_pass"]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# max_depth=4 ,min_samples_leaf= 100,

clf = LogisticRegression(C=1, penalty='l1')
clf = clf.fit(X_train, y_train)

train_model_y_score = [x[1] for x in clf.predict_proba(X_train)]
train_model_y = [int(x > 0.1) for x in train_model_y_score]

model_y_score = [x[1] for x in clf.predict_proba(X_test)]
model_y = [int(x > 0.1) for x in model_y_score]

print("roc_auc")
print(roc_auc_score(y_test, model_y))
print(roc_auc_score(y_train, train_model_y))
print("accuracy")
print(accuracy_score(y_test, model_y))
print(accuracy_score(y_train, train_model_y))
print("precision")
print(precision_score(y_test, model_y))
print(precision_score(y_train, train_model_y))
print("recall")
print(recall_score(y_test, model_y))
print(recall_score(y_train, train_model_y))
s = pickle.dumps(clf)
f = open('test_model.pkl', 'wb')
f.write(s)
f.close()