import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score

# get data
df_in = pd.read_csv('data/Smarket_ISLRinR.csv', header=0)
df_in.drop(columns='Unnamed: 0', inplace =True)
print(df_in.head(5))
print(df_in.columns)


