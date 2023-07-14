#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 07:52:42 2023

@author: gonzalo
"""


import numpy as np
import pandas as pd
from rdkit import Chem
import pickle
from datetime import datetime
from SLaux import calc_prop, one_hot_encode, SL_pred, SL_pred_prob

start_time = datetime.now()

props = ["tpsa","logp","rb","hbd","hba","mw","qed","nring","naring","fsp3"]


## ENSURE THE COMPOUND CLASSES AND IONIC CLASSES ARE WITHIN ANY OF THESE
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


path = ""

# Loading the data
SLmods = pickle.load(open(path + "SLgutper.sav", 'rb'))

base_models = SLmods[0]
meta_model = SLmods[1]
scaler = SLmods[2]


examp_comps = pd.DataFrame({
    "inchi": ['InChI=1S/C15H17NO10/c17-7-4-2-1-3-6(7)13(22)16-5-8(18)25-15-11(21)9(19)10(20)12(26-15)14(23)24/h1-4,9-12,15,17,19-21H,5H2,(H,16,22)(H,23,24)/t9-,10-,11+,12-,15+/m0/s1',
     'InChI=1S/C15H19NO2/c1-2-15(17)16-9-10-8-13(10)11-4-3-5-14-12(11)6-7-18-14/h3-5,10,13H,2,6-9H2,1H3,(H,16,17)/t10-,13+/m0/s1'],
    "ccl": ['Organic oxygen compounds', 'Organoheterocyclic compounds'],
    "icl": ['Acid', 'Neutral']})

examp_comps["mol"] = examp_comps.inchi.apply(lambda x: Chem.MolFromInchi(x))
examp_comps[props] = examp_comps.apply(lambda x: pd.Series(calc_prop(x["mol"])), axis = 1)
oh_ccl = one_hot_encode(examp_comps.ccl, comp_classes)
oh_icl = one_hot_encode(examp_comps.icl, ion_classes)
X = np.hstack([examp_comps[props].to_numpy(dtype = "float32"), oh_ccl, oh_icl])
gutper_pred = SL_pred(X, base_models, meta_model)
gutper_pred_prob = SL_pred_prob(X, base_models, meta_model)


## End time
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))