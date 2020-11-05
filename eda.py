"""
Final Project
EDA
"""


import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
import numpy as np
import seaborn as sns
from mlxtend.plotting import heatmap



df = pd.read_csv('student-mat-edited.csv')


df['school'] = df['school'].replace(['GP', 'MS'], [1, 0])
df['sex'] = df['sex'].replace(['M', 'F'], [1, 0])
df['address'] = df['address'].replace(['U', 'R'], [1, 0])
df['famsize'] = df['famsize'].replace(['GT3', 'LE3'], [1, 0])
df['Pstatus'] = df['Pstatus'].replace(['T', 'A'], [1, 0])
df['schoolsup'] = df['schoolsup'].replace(['yes', 'no'], [1, 0])
df['famsup'] = df['famsup'].replace(['yes', 'no'], [1, 0])
df['paid'] = df['paid'].replace(['yes', 'no'], [1, 0])
df['activities'] = df['activities'].replace(['yes', 'no'], [1, 0])
df['nursery'] = df['nursery'].replace(['yes', 'no'], [1, 0])
df['higher'] = df['higher'].replace(['yes', 'no'], [1, 0])
df['internet'] = df['internet'].replace(['yes', 'no'], [1, 0])
df['romantic'] = df['romantic'].replace(['yes', 'no'], [1, 0])

df = pd.get_dummies(df, prefix= ['Mjob', 'Fjob', 'reason', 'guardian'])

df['scores'] = df[['G1', 'G2', 'G3']].mean(axis=1)

df.info()
print(df.head())
