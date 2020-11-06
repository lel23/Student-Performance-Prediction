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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import sys
from sklearn.model_selection import train_test_split



df = pd.read_csv('student-mat-edited.csv')


df['school'] = df['school'].replace(['GP', 'MS'], [1, 0])
df['sex'] = df['sex'].replace(['M', 'F'], [1, 0])
df['address'] = df['address'].replace(['U', 'R'], [1, 0])
df['famsize'] = df['famsize'].replace(['GT3', 'LE3'], [1, 0])
df['Pstatus'] = df['Pstatus'].replace(['T', 'A'], [1, 0])
df = df.replace(to_replace={'yes':1, 'no':0})

df = pd.get_dummies(df, prefix= ['Mjob', 'Fjob', 'reason', 'guardian'])

#code from: https://stackoverflow.com/questions/46168450/replace-a-specific-range-of-values-in-a-pandas-dataframe
#convert the scores to integers representing the letter grade range specified in the paper. higher the number, the higher the grade
df['scores'] = df[['G1', 'G2', 'G3']].mean(axis=1)
df['scores'] = np.where(df['scores'].between(0, 10), 0, df['scores'])
df['scores'] = np.where(df['scores'].between(10, 12), 1, df['scores'])
df['scores'] = np.where(df['scores'].between(12, 14), 2, df['scores'])
df['scores'] = np.where(df['scores'].between(14, 16), 3, df['scores'])
df['scores'] = np.where(df['scores'].between(16, 21), 4, df['scores'])
df['scores'] = df['scores'].astype(np.int)

df = df.drop(index=1, columns=['G1', 'G2', 'G3'])

#separate into features and target
X = df[[i for i in list(df.columns) if i != 'scores']]
y = df['scores']
feat_labels = X.columns

#PCA#########

X_train, X_test, y_train, y_test =     train_test_split(X, y, test_size=0.3, 
                     stratify=y,
                     random_state=0)


stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

cov_mat = np.cov(X_train_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i])
               for i in range(len(eigen_vals))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k: k[0], reverse=True)


w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
               eigen_pairs[1][1][:, np.newaxis]))


# plt.savefig('images/05_03.png', dpi=300)
tot = sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.bar(range(1, len(df.columns)), var_exp, alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(range(1, len(df.columns)), cum_var_exp, where='mid',
         label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
# plt.savefig('images/05_02.png', dpi=300)
plt.show()
#print('Matrix W:\n', w)


X_train_std[0].dot(w)

X_train_pca = X_train_std.dot(w)
colors = ['r', 'b', 'g', 'y']
markers = ['s', 'x', 'o', '^']

for l, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=l, marker=m)

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
# plt.savefig('images/05_03.png', dpi=300)
plt.show()

#############

#random forest feature selection

stdsc = StandardScaler()
X = stdsc.fit_transform(X)

forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X, y)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

plt.title('Feature Importance')
plt.bar(range(X.shape[1]), 
        importances[indices],
        align='center')

plt.xticks(range(X.shape[1]), 
           feat_labels[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.tight_layout()
plt.savefig("rf_selection.png")
plt.show()

sfm = SelectFromModel(forest, prefit=True)
X_selected = sfm.transform(X)
print('Number of features that meet this threshold criterion:', 
      X_selected.shape[1])
print("Threshold %f" % np.mean(importances))


# Now, let's print the  features that met the threshold criterion for feature selection that we set earlier (note that this code snippet does not appear in the actual book but was added to this notebook later for illustrative purposes):

for f in range(X_selected.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

#df.info()
#print(df.head())
