import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTEENN
import pickle
import re
import subprocess
from tqdm import tqdm 

def random_forest(train_filepath, test_filepath, model_performance):
    x = pd.read_csv(train_filepath)
    y = pd.read_csv(test_filepath)

    sm = SMOTEENN()
    x, y = sm.fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=626)
    rcf = RandomForestClassifier()

    param_dist = {
        'n_estimators': np.arange(100, 626, 10),
        'max_depth': [None] + list(np.arange(10, 110, 10)),
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 11)
    }

    random_search = RandomizedSearchCV(estimator=rcf, param_distributions=param_dist, n_iter=7, cv=5, refit=True, verbose = 3)
    random_search.fit(x_train, y_train.values.ravel())
    pred = random_search.predict(x_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    model_performance.loc['Random Forest'] = ['Random Forest', accuracy, precision, recall, f1]
    pickle.dump(random_search, open('new_models/new-random_forest.pkl', 'wb'))
    return model_performance
