# coding : utf-8

import pandas as pd
import os

os.chdir("/Users/edouardcuny/Desktop/witlike/wit")
print(os.getcwd())
test = pd.read_excel("train_intent.xlsx", sep = ";")
test["pred"] = test["phrase"].apply(lambda x : x + "edouard")
print(test)
