import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from imblearn.combine import SMOTEENN
import pickle

def tensorflow_nn(train_filepath, test_filepath, model_performance):
    x = pd.read_csv(train_filepath)
    y = pd.read_csv(test_filepath)

    sm = SMOTEENN()
    x, y = sm.fit_resample(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=626)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=256, activation="relu", input_shape=x_train.shape[1:]),
        tf.keras.layers.Dense(units=72, activation="relu"),
        tf.keras.layers.Dense(units=1, activation="sigmoid")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy"])
    callbacks = tf.keras.callbacks.TensorBoard(log_dir='./logs')
    model.fit(x_train, y_train,  batch_size=32, epochs=100, shuffle=True, callbacks=[callbacks])
    pred = model.predict(x_test)

    pred_binary = (pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, pred_binary)
    precision = precision_score(y_test, pred_binary)
    recall = recall_score(y_test, pred_binary)
    f1 = f1_score(y_test, pred_binary)
    model_performance.loc['Tensorflow Neural Networks'] = ['Tensorflow Neural Networks', accuracy, precision, recall, f1]
    pickle.dump(model, open('new_models/new-tensorflow_nn.pkl', 'wb'))
    return model_performance