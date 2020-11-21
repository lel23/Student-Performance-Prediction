from collections import Counter
from confusion_matrix import plot_confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler
from sklearn.utils import resample
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
       'nursery', 'higher', 'internet', 'romantic', 'famrel', 'freetime',
       'goout', 'Dalc', 'Walc', 'health', 'absences', 'Mjob_services',
       'Mjob_teacher', 'Fjob_at_home', 'Fjob_services', 'Fjob_teacher',
       'reason_home', 'reason_other', 'guardian_father', 'guardian_mother']


#list of features in order of importance determined by RF feature selection
rf_features = ['reason_home', 'Fjob_at_home', 'age', 'Fjob_other',
               'Mjob_at_home', 'Mjob_other', 'Fedu', 'activities',
               'failures', 'internet', 'address', 'health', 'freetime',
               'Walc', 'Dalc', 'nursery', 'reason_course', 'Fjob_teacher',
               'romantic', 'paid', 'traveltime', 'famsize', 'studytime',
               'sex', 'Pstatus', 'Medu', 'Fjob_health', 'reason_reputation',
               'schoolsup', 'famrel', 'Fjob_services', 'guardian_father',
               'scores', 'famsup', 'Mjob_services', 'guardian_mother',
               'higher', 'Mjob_teacher', 'guardian_other', 'Mjob_health',
               'absences', 'goout', 'reason_other']    
  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state=0,
                                                    stratify=y)
#standardizing the dataset                     
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


### Feature Extraction ###
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



# Grid search to determine which hyper parameters are best
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 25, 50, 75, 100.0, 1000.0]
solvers = ['newton-cg', 'sag', 'saga', 'lbfgs']
param_grid = [{'penalty':['l2'], 'C':param_range, 'solver':solvers + ['liblinear']},
              {'penalty':['none'], 'solver':solvers}]

gs = GridSearchCV(estimator=LogisticRegression(class_weight='balanced', max_iter=10000), 
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


gs = GridSearchCV(estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
                param_grid=param_grid,
                scoring='accuracy',
                cv=2)
scores = cross_val_score(gs, X_train, y_train,
                        scoring='accuracy', cv=5)
print("\n\nCV Accuracy: %.3f +/- %.3f" % (np.mean(scores), np.std(scores)))

# Hyper parameters determined from previous grid search
lr = LogisticRegression(penalty='l2', C=.001, solver='newton-cg', max_iter=1000)

#Calculate accuracy, precision, recall, and f1-score using each RF and SBS feature selection
num_features = list(range(3, 44))
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
       
        X_train_norm = norm.fit_transform(X_train_new)
        X_test_norm = norm.transform(X_test_new)

        X_train_lda = lda.fit_transform(X_train_norm, y_train)
        X_test_lda = lda.transform(X_test_norm)

        print("\n\nNumber of features:", num)

        lr.fit(X_train_lda, y_train)
        y_pred = lr.predict(X_test_lda)

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
    plt.savefig(name+"_fs_metrics_logreg.png")
    plt.show()


#Create final model and confusion matrix
X_train_new = X_train[X_train.columns.intersection(rf_features)]
X_test_new = X_test[X_test.columns.intersection(rf_features)]

X_train_std = stdsc.fit_transform(X_train_new)
X_test_std = stdsc.transform(X_test_new)

lr.fit(X_train_std, y_train)
y_pred = lr.predict(X_test_std)

print("Metrics for final model:\n\n")

accuracy = accuracy_score(y_pred, y_test)
precision = precision_score(y_pred, y_test, average='weighted')
recall = recall_score(y_pred, y_test, average='weighted')
f1 = f1_score(y_pred, y_test, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)


cnf_matrix = confusion_matrix(y_test, y_pred, labels=list(range(5)))
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['F', 'D', 'C', 'B', 'A'], title='Logistic Regression Confusion Matrix')
plt.savefig('confusion_matrix_logreg.png')
plt.show()
