# gutmetabos
Source code and data for Super Learner model to predict gut permanence

Better use a conda environment like this:

conda create -n gutper -c conda-forge rdkit jupyter pandas numpy spyder scikit-learn scipy seaborn

To get predictions: edit with your compounds and use runSL.py

It uses an already trained model saved in SLgutper.sav

To re-train the model: use SLtrainer.py

gutper_set2.csv contains the training dataset
