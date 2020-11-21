import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from collections import Counter

from imblearn.over_sampling import SMOTE

import sys

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

#https://machinelearningmastery.com/multi-class-imbalanced-classification/
#Get rid of class imbalance in the target feature by oversampling
oversample = SMOTE(random_state=0)
X, y = oversample.fit_resample(X, y)



#list of features in order of importance determined by SBS feature selection
sbs_features = ['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu',
       'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid',
       'activities', 'higher', 'internet', 'romantic', 'famrel', 'freetime',
       'goout', 'Dalc', 'health', 'Mjob_at_home', 'Mjob_health', 'Mjob_other',
       'Mjob_services', 'Fjob_at_home', 'Fjob_teacher', 'reason_course',
       'reason_home', 'reason_other', 'reason_reputation', 'guardian_mother']


#list of features in order of importance determined by RF feature selection
rf_features = ['absences', 'health', 'Medu', 'freetime', 'age', 'goout', 'famrel', 
               'Fedu', 'Walc', 'studytime', 'nursery', 'failures', 'Mjob_other',                      
               'famsup', 'paid', 'Fjob_other', 'sex', 'famsize', 'guardian_mother', 
               'activities', 'traveltime', 'Dalc', 'romantic', 'reason_course', 'internet',
               'Mjob_services', 'address', 'Pstatus', 'reason_reputation', 'schoolsup',
               'Fjob_services', 'guardian_father', 'reason_home', 'school', 'Mjob_teacher', 
               'Mjob_at_home', 'Mjob_health', 'Fjob_teacher', 'reason_other', 'Fjob_at_home', 
               'higher', 'guardian_other', 'Fjob_health']      
         



#split into testing and training set
X_train, X_test, y_train, y_test =    train_test_split(X, y, 
                     test_size=0.3, 
                     random_state=0, 
                     stratify=y)

#standardizing the dataset                     
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)



#Feature Extraction

#PCA
scikit_pca = PCA(n_components=2)
X_train_pca = scikit_pca.fit_transform(X_train_std)
X_test_pca = scikit_pca.fit_transform(X_test_std)


#KPCA
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_train_kpca = scikit_kpca.fit_transform(X_train_std)
X_test_kpca = scikit_kpca.fit_transform(X_test_std)


#LDA

#normalize the dataset before performing LDA   


norm = Normalizer()
X_train_norm = norm.fit_transform(X_train)
X_test_norm = norm.transform(X_test)


lda = LDA(n_components=3)
X_train_lda = lda.fit_transform(X_train_norm, y_train)
X_test_lda = lda.transform(X_test_norm)



#Grid search to determine which hyper parameters are best

param_range = [3, 4, 5, 6]


param_grid = [{'max_depth':param_range,
               'criterion':['gini', 'entropy'],
               'random_state':range(0,30)}]


gs = GridSearchCV(estimator=DecisionTreeClassifier(class_weight='balanced'), 
                  param_grid=param_grid,
                  scoring='accuracy',
                  refit=True,
                  cv=10,
                  n_jobs=-1)


#grid search for each feature extraction method
for X_train_gs, name in zip([X_train_std, X_train_lda, X_train_pca, X_train_kpca], ["No Feature Extraction", "LDA", "PCA", "KPCA"]):
    print("\n\nGrid search for", name)
    gs = gs.fit(X_train_gs, y_train)
    print("Accuracy:", gs.best_score_)
    print("Parameters:", gs.best_params_)

gs = GridSearchCV(estimator=DecisionTreeClassifier(class_weight='balanced'),
                param_grid=param_grid,
                scoring='accuracy',
                cv=2)

scores = cross_val_score(gs, X_train, y_train,
                        scoring='accuracy', cv=5)

print("\n\nCV Accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))


# Hyper parameters determined from previous grid search
tree = DecisionTreeClassifier(criterion  = 'gini', max_depth = 6, random_state = 8)

#Calculate accuracy, precision, recall, and f1-score using each RF and SBS feature selection
num_features = list(range(2, 44))
for name, features in zip(["RF", "SBS"], [rf_features, sbs_features]):
    print("\n\n" + name + " Feature Selection")
    accuracy_list = []
    precision_list = [] 
    recall_list = []
    f1_list = []

    for num in num_features:
        #Code from https://stackoverflow.com/questions/40636514/selecting-pandas-dataframe-column-by-list
        X_train_new = X_train[X_train.columns.intersection(features[:num])]
        X_test_new = X_test[X_test.columns.intersection(features[:num])]
       
        X_train_std = stdsc.fit_transform(X_train_new)
        X_test_std = stdsc.transform(X_test_new)

        print("\n\nNumber of features:", num)

        tree.fit(X_train_std, y_train)
        y_pred = tree.predict(X_test_std)

        accuracy = accuracy_score(y_pred, y_test)
        precision = precision_score(y_pred, y_test, average='weighted')
        recall = recall_score(y_pred, y_test, average='weighted')
        f1 = f1_score(y_pred, y_test, average='weighted')

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    #plot metrics for each feature selection method
    plt.plot(num_features, accuracy_list, label='accuracy')
    plt.plot(num_features, precision_list, label='precision')
    plt.plot(num_features, recall_list, label='recall')
    plt.plot(num_features, f1_list, label='f1-score')

    plt.title(name)
    plt.xlabel('Number of Features Used')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(name+"_fs_metrics_decisiontree.png")
    plt.show()
    
