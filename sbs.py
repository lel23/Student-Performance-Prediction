import pandas as pd
from io import StringIO
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.base import clone
from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



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

X = df[[i for i in list(df.columns) if i != "scores"]]
y = df['scores']

feat_labels = X.columns

oversample = SMOTE()
X, y = oversample.fit_resample(X, y)


X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)



# # # Bringing features onto the same scale

mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_test_norm = mms.transform(X_test)

stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)



class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        
        X_train, X_test, y_train, y_test =             train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, 
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X_train, y_train, 
                                         X_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

#(Sam) I used SVM for the estimator because I am working on the SVM implementation, but I recommend using your type of model before running this
svm = SVC(C=10, kernel='rbf', gamma=.1)

# selecting features

sbs = SBS(svm, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.1, 1.02])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.tight_layout()
plt.savefig("sbs.png")
plt.show()

k3 = list(sbs.subsets_[10])
print(df.columns[1:][k3])

svm.fit(X_train_std, y_train)
print('Training accuracy:', svm.score(X_train_std, y_train))
print('Test accuracy:', svm.score(X_test_std, y_test))

svm.fit(X_train_std[:, k3], y_train)
print('Training accuracy:', svm.score(X_train_std[:, k3], y_train))
print('Test accuracy:', svm.score(X_test_std[:, k3], y_test))
