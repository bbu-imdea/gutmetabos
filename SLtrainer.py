#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 11:47:32 2023

@author: gonzalo
"""

import numpy as np
import pandas as pd
from rdkit import Chem
import random
import pickle
from sklearn.model_selection import StratifiedKFold
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from SLaux import calc_prop, one_hot_encode, oof_pred

start_time = datetime.now()
random.seed(42)


props = ["tpsa","logp","rb","hbd","hba","mw","qed","nring","naring","fsp3"]
comp_classes = ['Benzenoids',
 'Endocannabinoids',
 'Fatty Acyls',
 'Glycerolipids',
 'Glycerophospholipids',
 'Hydrocarbons',
 'Nucleosides, nucleotides, and analogues',
 'Organic acids and derivatives',
 'Organic nitrogen compounds',
 'Organic oxygen compounds',
 'Organoheterocyclic compounds',
 'Organosulfur compounds',
 'Other',
 'Phenylpropanoids and polyketides',
 'Prenol lipids',
 'Sphingolipids',
 'Steroids and steroid derivatives']
ion_classes = ['Acid', 'Basic', 'Neutral', 'Zwitterion']


"""
#####################################################
########## EXTRACT 7 FOLDS FOR TRAINING #############
#####################################################
"""  

# Loading the data
path = ""
fname = 'gutper_set2.csv'
df = pd.read_csv(path + fname, sep = ';')
df = df.dropna()
df["mol"] = df.inchi.apply(lambda x: Chem.MolFromInchi(x))
df.ccl.value_counts()
df[props] = df.apply(lambda x: pd.Series(calc_prop(x["mol"])), axis = 1)
oh_ccl = one_hot_encode(df.ccl, comp_classes)
oh_icl = one_hot_encode(df.icl, ion_classes)
X = np.hstack([df[props].to_numpy(dtype = "float32"), oh_ccl, oh_icl])
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
y = df.gutper.to_numpy()

k = 8
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
selected_indices = [(train_index, test_index) for train_index, test_index in skf.split(df.mol, df.ccl)]

X = X[selected_indices[0][0]]
y = y[selected_indices[0][0]]
df_cv = df.iloc[selected_indices[0][0]]    
df_cv.reset_index(inplace = True, drop = True)


"""
#####################################################
############### TRAIN SUPER LEARNER #################
#####################################################
"""  

# list of base models
rs = 12
base_models = list()
base_models.append(LogisticRegression(solver='liblinear'))
base_models.append(DecisionTreeClassifier(random_state = rs))
base_models.append(SVC(gamma='scale', probability=True))
base_models.append(GaussianNB())
base_models.append(KNeighborsClassifier())
base_models.append(AdaBoostClassifier(random_state = rs))
base_models.append(BaggingClassifier(n_estimators=10, random_state = rs))
base_models.append(RandomForestClassifier(n_estimators=10, random_state = rs))
base_models.append(ExtraTreesClassifier(n_estimators=10, random_state = rs))

# Oof preds from cv
oof_X, oof_y = oof_pred(X, y, base_models, df_cv.ccl)

# Fit base models with full cv data
for model in base_models:
    model.fit(X, y)

## Fit metamodel
meta_model = LogisticRegression(solver='liblinear')
meta_model.fit(oof_X, oof_y)

## Save objects 
SLmods = [base_models, meta_model, scaler]
pickle.dump(SLmods, open(path + "SLgutper.sav", 'wb'))

## End time
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))