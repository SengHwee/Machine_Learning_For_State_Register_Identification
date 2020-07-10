import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import shutil
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from joblib import Parallel, delayed
from multiprocessing import pool

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import TensorBoard
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from mlxtend.evaluate import scoring


import talos as ta
# from talos.model.layers import hidden_layers
from talos.model.early_stopper import early_stopper
from talos import Scan
from talos import Deploy
import talos_model_accuracy_checker as checker

############ Feature Selection +  model training (mlxtend framework) ##########

def transpose_data(df):
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df.reset_index()
    df = df.drop(df.index[0])
    df = df.reset_index()
    df = df.drop(["level_0", "index"],
                 axis=1)
    return df


########## Loading training data, creation of directory and combine ##########
data_array = []
TIMESTR = time.strftime("%Y%m%d-%H%M%S")
TRAINDIR = 'csvs/train/'
DATADIR = 'data/runs_{}/'.format(TIMESTR)
TRAINED_MODEL_DIR = DATADIR+'model/'
if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)
if not os.path.exists(TRAINED_MODEL_DIR):
    os.makedirs(TRAINED_MODEL_DIR)


for train_data in os.listdir(TRAINDIR):
    df = pd.read_csv(TRAINDIR + train_data)
    df = transpose_data(df)
    data_array.append(df)

df_train = pd.concat(data_array)
df_train.to_csv(DATADIR+"combine_training_data.csv", index=False)
df_y_train = df_train.pop('xx_state_ff')
df_train = df_train.drop(['xx_good_SN'], axis=1)
df_train = df_train.drop(['has_feedback_path'], axis=1)
df_train = df_train.drop(['load_centrality'], axis=1)

# print(df_y_train)
# print(df_train)

################ load testing data ##################
target_name = 'xx_state_ff'
df_test = pd.read_csv('csvs/test/uart.csv')
df_test = transpose_data(df_test)
df_y_test = df_test.pop(target_name)
df_test = df_test.drop(['xx_good_SN'], axis=1)
df_test = df_test.drop(['has_feedback_path'], axis=1)
df_test = df_test.drop(['load_centrality'], axis=1)

# print(df_y_test)
# print(df_test)

feature_names = list(df_train.columns)

####### Pre-Train: Feature Selection - Filter method ##########


constant_filter = VarianceThreshold(threshold=0)  # remove columns with contstant values
quasi_constant_filter = VarianceThreshold(threshold=0.01)  # remove columns with 1% difference in values
constant_filter_df_train = constant_filter.fit_transform(df_train)
constant_filter_df_test = constant_filter.transform(df_test)


feature_names = list(df_train[df_train.columns[constant_filter.get_support(indices=True)]].columns)
df_train = constant_filter_df_train
df_test = constant_filter_df_test
feature_names_constant_filter = pd.DataFrame(feature_names)
feature_names_constant_filter.to_csv(DATADIR+"feature_names_constant_filter.csv", index=False)


################ scaling and converting of data ###################
scalar = StandardScaler()

df_train = scalar.fit_transform(df_train)
df_test = scalar.transform(df_test)

df_y_train = df_y_train.to_numpy().astype(int)
df_y_test = df_y_test.to_numpy().astype(int)


####################### model #####################################

def classification_model(hidden_layers, first_neuron, hidden_neurons, optimizer='adam', loss='binary_crossentropy'):
    model = Sequential()
    model.add(Dense(first_neuron, activation='elu'))  

    for _ in range(hidden_layers):
        model.add(Dense(hidden_neurons, input_shape= (df_train.shape[1],), activation='relu'))
    
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


keras_est = classification_model(hidden_layers=15, first_neuron=20, hidden_neurons=217, optimizer='RMSprop', loss='binary_crossentropy')
keras_est.fit(df_train, df_y_train, shuffle=False, epochs=22, batch_size=10)
df_y_pred = keras_est.predict_classes(df_test)
print(df_y_pred, df_y_test)
keras_est.evaluate(df_test, df_y_test)

true_positive, state_register_index, state_register_accuracy, accuracy =  checker.true_positive_checker(df_y_test, df_y_pred)
print(state_register_accuracy)
