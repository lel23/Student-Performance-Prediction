import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

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

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=1)



#feat_labels = random_forest_fs(X_train_std, y_train)
"""
Random Forest Feature Selection
"""
stdsc = Normalizer()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.fit_transform(X_test)
feat_labels = X.columns


forest = RandomForestClassifier(n_estimators=500,
                                random_state=1)

forest.fit(X, y)
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]

for f in range(X.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

sfm = SelectFromModel(forest, prefit=True)
X_selected = sfm.transform(X)
print('Number of features that meet this threshold criterion:', 
      X_selected.shape[1])
print("Threshold %f" % np.mean(importances))


# Now, let's print the  features that met the threshold criterion for feature selection that we set earlier (note that this code snippet does not appear in the actual book but was added to this notebook later for illustrative purposes):
cols = []
for f in range(X_selected.shape[1]):
    cols.append(feat_labels[indices[f]])    
    print("%2d) %-*s %f" % (f + 1, 30, 
                            feat_labels[indices[f]], 
                            importances[indices[f]]))

X_train_std = X_train_std[:, :X_selected.shape[1]]
X_test_std = X_test_std[: , :X_selected.shape[1]]







def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot examples by class
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    color=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)



# ## LDA via scikit-learn


lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std, y_train)



lr = LogisticRegression(multi_class='ovr', random_state=1, solver='lbfgs')
lr = lr.fit(X_train_lda, y_train)

plot_decision_regions(X_train_lda, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('lda_train.png', dpi=300)
plt.show()


X_test_lda = lda.transform(X_test_std)

plot_decision_regions(X_test_lda, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('lda_test.png', dpi=300)
plt.show()



