#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 09:52:00 2023

@author: gonzalo
"""

from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from numpy import hstack
from numpy import vstack
from numpy import asarray

def calc_prop(mol):
    tpsa = Descriptors.TPSA(mol)
    logp = Descriptors.MolLogP(mol)
    rb = Descriptors.NumRotatableBonds(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    mw = Descriptors.MolWt(mol)
    qed = Descriptors.qed(mol)
    nring = Chem.rdMolDescriptors.CalcNumRings(mol)
    naring = Chem.rdMolDescriptors.CalcNumAromaticRings(mol)
    fsp3 = Chem.rdMolDescriptors.CalcFractionCSP3(mol)
    return [tpsa, logp, rb, hbd, hba, mw, qed, nring, naring, fsp3]

def one_hot_encode(vector, categories):
    vector = np.array(vector).reshape(-1, 1)
    encoder = preprocessing.OneHotEncoder(categories = [categories], sparse_output = False, handle_unknown = 'ignore')
    encoder.fit(vector)
    one_hot_matrix = encoder.transform(vector)
    return one_hot_matrix

def oof_pred(X, y, models, ccl):
    oof_X, oof_y = list(), list()
    kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    for train_ix, test_ix in kfold.split(X, ccl):  
        y_preds = list()
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        oof_y.extend(test_y)
        for model in models:
            model.fit(train_X, train_y)
            y_pred = model.predict_proba(test_X)
            y_preds.append(y_pred)
        oof_X.append(hstack(y_preds))
    return vstack(oof_X), asarray(oof_y)

def SL_pred(X, models, meta_model):
 	meta_X = list()
 	for model in models:
         y_pred =  model.predict_proba(X)
         meta_X.append(y_pred)
 	meta_X = hstack(meta_X)
 	return meta_model.predict(meta_X)

def SL_pred_prob(X, models, meta_model):
    meta_X = list()
    for model in models:
         y_pred_prob = model.predict_proba(X)
         meta_X.append(y_pred_prob)
    meta_X = hstack(meta_X)
    return meta_model.predict_proba(meta_X)

