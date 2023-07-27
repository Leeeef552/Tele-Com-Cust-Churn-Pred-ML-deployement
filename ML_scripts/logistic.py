import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.combine import SMOTEENN
import pickle
from tqdm import tqdm

def logistic(train_filepath, test_filepath, model_performance):
    x = pd.read_csv(train_filepath)
    y = pd.read_csv(test_filepath)

    sm = SMOTEENN()
    x, y = sm.fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=626)
    lg = LogisticRegression(max_iter=1000)

    total_iterations = 10  # Total number of iterations (can be adjusted based on desired progress granularity)
    update_interval = 1     # Update progress every iteration (fine-grained progress)

    # Manually create the progress bar
    with tqdm(total=total_iterations, position=0, leave=True, desc="Training Logistic Regression") as pbar:
        for i in range(total_iterations):
            lg.fit(x_train, y_train.values.ravel())
            pbar.update(update_interval)

    pred = lg.predict(x_test)

    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    model_performance.loc['Logistic Regression'] = ['Logistic Regression', accuracy, precision, recall, f1]
    pickle.dump(lg, open('new_models/new-lg.pkl', 'wb'))
    return model_performance

