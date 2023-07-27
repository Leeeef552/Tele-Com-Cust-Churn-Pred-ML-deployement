import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTEENN
import pickle
from tqdm import tqdm

def knn(train_filepath, test_filepath, model_performance):
    x = pd.read_csv(train_filepath)
    y = pd.read_csv(test_filepath)

    sm = SMOTEENN()
    x, y = sm.fit_resample(x, y)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(x)
    scaled = pd.DataFrame(scaled, columns=x.columns)
    x_train, x_test, y_train, y_test = train_test_split(scaled, y, test_size=0.3, random_state=626)

    error_rate = []
    for i in tqdm(range(1, 40), desc = 'Training KNN'):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(x_train, y_train.values.ravel())
        pred_i = knn.predict(x_test)
        error_rate.append(np.mean(pred_i != y_test['Churn']))

    knn2 = KNeighborsClassifier(n_neighbors=error_rate.index(min(error_rate))+1)
    knn2.fit(x_train, y_train.values.ravel())
    pred = knn2.predict(x_test)
    
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    
    model_performance.loc['KNN'] = ['KNN',accuracy, precision, recall, f1]
    pickle.dump(knn2, open('new_models/new-KNN.pkl', 'wb'))

    return model_performance
