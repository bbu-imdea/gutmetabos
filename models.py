# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 09:36:46 2023

@author: gonzalo.colmenarejo
"""

path = "C:/Users/gonzalo.colmenarejo.PT-IMD-19/Documents/papers/METABOS/JMC/code and data"

import sys
sys.path.append(path)
import numpy as np
import pandas as pd
from datetime import datetime
#•import funcanalysis as fa
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestClassifier
# import random
import pickle


#import phfisher.py as ph

start_time = datetime.now()

pchpros = ["tpsa","logp","rb","hbd","hba","mw","qed","nring","naring","fsp3"]
comp_order = ['Organoheterocyclic compounds',
  'Glycerolipids',
  'Benzenoids',
  'Organic acids and derivatives',
  'Organic oxygen compounds',
  'Other',
  'Steroids and steroid derivatives',
  'Fatty Acyls',
  'Phenylpropanoids and polyketides',
  'Prenol lipids',
  'Glycerophospholipids',
  'Organic nitrogen compounds',
  'Nucleosides, nucleotides, and analogues',
  'Organosulfur compounds',
  'Hydrocarbons',
  'Sphingolipids',
  'Saccharolipids',
  'Endocannabinoids']
comp_order.remove("Saccharolipids") # Removed as it is not in gut metablites

scratch = True # Regenerate vs reuse models from scratch


def get_iclass(x):
    ac = True if ~pd.isna(x["pkasa"]) and x["pkasa"] <= 7.4 else False
    ba = True if ~pd.isna(x["pkasb"]) and x["pkasb"] > 7.4 else False
    if ac is False and ba is False:
        icl = "Neutral"
    else: 
        if ac is False and ba is True:
            icl = "Basic"
        else:
            if ac is True and ba is False:
                icl = "Acid"
            else:
                icl = "Zwitterion"
    return icl


def get_iqr(d, cset, comp_order, pchpros):
    iqr = pd.DataFrame(columns = [y for l in [["l-" + x, "u-" + x] for x in pchpros] for y in l]+["ic1","ic2"], index=range(len(comp_order)))
    iqr.index = comp_order
    for i in range(len(pchpros)):
        for j in range(len(comp_order)):
            pro = pchpros[i]
            cs = comp_order[j]
            dat = d.loc[(d.ccl == cs) & (d.set == cset), pro]
            iqr.loc[cs, "l-"+pro] = np.nanpercentile(dat, 25)
            iqr.loc[cs, "u-"+pro] = np.nanpercentile(dat, 75)
    for j in range(len(comp_order)):
        cs = comp_order[j]
        dat = d.loc[(d.ccl == cs) & (d.set == cset), "icl"]
        iqr.loc[cs, "ic1"] = dat.value_counts().index[0]
        if len(dat.value_counts().index) > 1:   
            iqr.loc[cs, "ic2"] = dat.value_counts().index[1]
        else:
            iqr.loc[cs, "ic2"] = dat.value_counts().index[0]
    iqr = iqr.dropna().copy()        
    return iqr


def check_guy(iqr, pchpros, x):
    if x["ccl"].iloc[0] in iqr.index.tolist():
        iqr = iqr[iqr.index == x["ccl"].iloc[0]].copy()
        #x_icl = x["icl"]
        x = x[pchpros].copy()
        x.reset_index(drop = True, inplace = True)
        iqr_l = iqr.loc[:,iqr.columns.to_series().apply(lambda x: "l-" in x)]
        iqr_l.columns = pchpros
        iqr_l.reset_index(drop = True, inplace = True)
        iqr_u = iqr.loc[:,iqr.columns.to_series().apply(lambda x: "u-" in x)]
        iqr_u.columns = pchpros
        iqr_u.reset_index(drop = True, inplace = True)
        #out = ((x <= iqr_u) & (x >= iqr_l)).iloc[0,:].sum() + int(x_icl.iloc[0] in iqr[["ic1"]].iloc[0,:].tolist())
        out = ((x <= iqr_u) & (x >= iqr_l)).iloc[0,:].sum()
    else:
        out = np.nan
    return out


def get_premed(d, comp_order):
    acc = d[(d.set == d.pset)].shape[0]/d.shape[0]
    pre = d[(d.set == d.pset) & (d.pset == "gut")].shape[0]/d[(d.pset == "gut")].shape[0]
    rec = d[(d.set == d.pset) & (d.set == "gut")].shape[0]/d[(d.set == "gut")].shape[0]  
    acc_ccl = np.empty(shape = (len(comp_order)))
    acc_ccl[:] = np.nan
    pre_ccl = np.empty(shape = (len(comp_order)))
    pre_ccl[:] = np.nan
    rec_ccl = np.empty(shape = (len(comp_order)))
    rec_ccl[:] = np.nan
    for k in range(len(comp_order)):
        comps = comp_order[k]
        if d[(d.ccl == comps)].shape[0] > 0:
            acc_ccl[k] = d[(d.set == d.pset) & (d.ccl == comps)].shape[0]/d[(d.ccl == comps)].shape[0]
        if d[(d.pset == "gut") & (d.ccl == comps)].shape[0] > 0:
            pre_ccl[k] = d[(d.set == d.pset) & (d.pset == "gut") & (d.ccl == comps)].shape[0]/d[(d.pset == "gut") & (d.ccl == comps)].shape[0]
        if d[(d.set == "gut") & (d.ccl == comps)].shape[0] > 0:
            rec_ccl[k] = d[(d.set == d.pset) & (d.set == "gut") & (d.ccl == comps)].shape[0]/d[(d.set == "gut") & (d.ccl == comps)].shape[0]
    return acc, pre, rec, acc_ccl, pre_ccl, rec_ccl


def score_pred(d, iqrs, iqrg, mdg, mds, pchpros = pchpros, comp_order = comp_order):
    d.loc[:,"hitg"] = d.apply(lambda x: check_guy(iqr = iqrg, pchpros = pchpros, x = x.to_frame().T), axis = 1)
    d.loc[:,"hits"] = d.apply(lambda x: check_guy(iqr = iqrs, pchpros = pchpros, x = x.to_frame().T), axis = 1)
    d.loc[:,"mseg"] = d.apply(lambda x: mse(x[pchpros].to_frame().T, mdg[mdg.index == x["ccl"]]) if x["ccl"] in mdg.index else np.nan, axis = 1)
    d.loc[:,"mses"] = d.apply(lambda x: mse(x[pchpros].to_frame().T, mds[mds.index == x["ccl"]]) if x["ccl"] in mds.index else np.nan, axis = 1)
    d.loc[:,"pset"] = d.apply(lambda x: "serum" if x["hits"] > x["hitg"] else "gut" if x["hits"] < x["hitg"] else "serum" if 
                                           x["mses"] < x["mseg"] else "gut" if x["mses"] > x["mseg"] else "u", axis = 1)
    return d





if scratch == True:
    ## Regenerate dataset
    full = pd.read_csv(path + "/full.csv", sep = ";", low_memory = False)
    #full["mol"] = full.molblock.apply(lambda x: fa.mb2mol(x))
    full["icl"] = full.apply(lambda x: get_iclass(x), axis = 1)
    #full["pch"] = full.pchar.apply(lambda x: 0 if x == 0 else "-" if x < 0 else "+")
    full2 = full[full.set.isin(["serum","gut"])]
    full2.reset_index(drop = True, inplace = True)
    full2.set.value_counts(dropna = False)
    # serum    16621
    # gut       5021
    full2_st = ((full2[pchpros]-full2[pchpros].mean())/full2[pchpros].std()).copy()
    full2_st[["icl","ccl","set"]] = full2[["icl","ccl","set"]]
    full2_st = full2_st[full2_st.ccl.isin(comp_order)].copy()
    full2_st.reset_index(drop = True, inplace = True)
    full2_st.to_csv(path + "/full2_st.csv", sep = ";", index = False)
    
    ## SCORE METHOD: regenerate iqrs and medians
    iqrs = get_iqr(full2_st, cset ="serum", comp_order = comp_order, pchpros = pchpros)
    iqrg = get_iqr(full2_st, cset = "gut", comp_order = comp_order, pchpros = pchpros)
    mds = full2_st[full2_st.set == "serum"].groupby("ccl", as_index = True).apply(lambda x: x[pchpros].median().to_frame().T)
    mds.reset_index(level =1, inplace = True, drop = True)
    mdg = full2_st[full2_st.set == "gut"].groupby("ccl", as_index = True).apply(lambda x: x[pchpros].median().to_frame().T)
    mdg.reset_index(level =1, inplace = True, drop = True)
    iqrs.to_csv(path + "/iqrs.csv", sep = ";", index = True)
    iqrg.to_csv(path + "/iqrg.csv", sep = ";", index = True)
    mds.to_csv(path + "/mds.csv", sep = ";", index = True)
    mdg.to_csv(path + "/mdg.csv", sep = ";", index = True)
    

    ## ML METHOD: train the model
    x_train = pd.get_dummies(full2_st[pchpros+["icl","ccl"]])
    y_train = full2_st["set"]     
    rf = RandomForestClassifier(n_estimators = 1000, n_jobs = -1, random_state = 1234)
    rf.fit(x_train, y_train)    
    pickle.dump(rf, open(path + "/rf", 'wb'))
else:
    full2_st = pd.read_csv(path + "/full2_st.csv", sep = ";", low_memory = False)
    iqrs = pd.read_csv(path + "/iqrs.csv", sep = ";", low_memory = False, index_col = 0)
    iqrg = pd.read_csv(path + "/iqrg.csv", sep = ";", low_memory = False, index_col = 0)
    mds = pd.read_csv(path + "/mds.csv", sep = ";", low_memory = False, index_col = 0)
    mdg = pd.read_csv(path + "/mdg.csv", sep = ";", low_memory = False, index_col = 0)
    rf = pickle.load(open(path + "/rf", 'rb'))


## Example usage (with full2 set)
d = full2_st.copy()
d = score_pred(d = d, iqrs = iqrs, iqrg = iqrg, mds = mds, mdg = mdg).copy()
acc, pre, rec, acc_ccl, pre_ccl, rec_ccl = get_premed(d, comp_order)
f1 = 2*pre*rec/(pre+rec)
print("acc: %.2f"%acc, "pre: %.2f"%pre, "rec: %.2f"%rec, "f1: %.2f"%f1)

d = full2_st.copy()
x_train = pd.get_dummies(d[pchpros+["icl","ccl"]])
y_train = d["set"]     
d["pset"] = rf.predict(x_train)
acc, pre, rec, acc_ccl, pre_ccl, rec_ccl = get_premed(d, comp_order)
f1 = 2*pre*rec/(pre+rec)
print("acc: %.2f"%acc, "pre: %.2f"%pre, "rec: %.2f"%rec, "f1: %.2f"%f1)


# End time counter
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
