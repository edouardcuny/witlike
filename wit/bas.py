# coding : utf-8

import json
import pandas as pd
import os

os.chdir("/Users/edouardcuny/Desktop/witlike/wit")
train = pd.read_excel("train_intent_only_intents.xlsx", sep = ";")
x_train = train.ix[:, 0]
x_train = x_train.tolist()
y_train = train.ix[:, 1]
y_train = y_train.tolist()

print(json.dumps( {{"x_train" : x_train},{"y_train" : y_train}}))
