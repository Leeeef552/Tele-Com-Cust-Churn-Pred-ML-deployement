import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTEENN
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV

def SVM(train_filepath, test_filepath, model_performance):
    x = pd.read_csv(train_filepath)
    y = pd.read_csv(test_filepath)

    sm = SMOTEENN()
    x, y = sm.fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=626)
    svc = SVC()
    para2 = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'auto', 'scale'], 
              'kernel': ['rbf']}
    random_search = RandomizedSearchCV(estimator=svc, param_distributions=para2, n_iter=7, cv=5, refit=True, verbose=3)
    random_search.fit(x_train, y_train.values.ravel())
    pred = random_search.predict(x_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    model_performance.loc['SVM'] = ['SVM', accuracy, precision, recall, f1]
    pickle.dump(random_search, open('new_models/new-svm.pkl', 'wb'))
    return model_performance
