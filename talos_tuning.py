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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import TensorBoard


import talos as ta
# from talos.model.layers import hidden_layers
from talos.model.early_stopper import early_stopper
from talos import Scan
from talos import Deploy
from talos.utils.gpu_utils import force_cpu


############# FUNCTIONS ####################
#transpose, reorder and drop
def transpose_data(df):
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df.reset_index()
    df = df.drop(df.index[0])
    df = df.reset_index()
    df = df.drop(["level_0", "index"],
                 axis=1)
    return df

#Creation of Model Function
def binary_model (x_train, y_train, x_val, y_val, params):
    ########### Tracking performance on tensorboard ####################
    # tensorboard = TensorBoard(log_dir = TRAINED_MODEL_DIR+'logs/{}'.format(params))

    print(params)
    model = Sequential()

    #initial layer
    model.add(Dense(params['first_neuron'], input_shape=(x_train.shape[1],), activation=params['activation']))
    model.add(Dropout(params['dropout']))

    #hidden layers
    for i in range(params['hidden_layers']):
        # print(f"adding layer {i+1}")
        model.add(Dense(params['hidden_neurons'], activation = params['activation']))
        model.add(Dropout(params['dropout']))

    #final layer
    if params['loss'] == 'binary_crossentropy':
        model.add(Dense(1, activation = params['last_activation']))
    else:
        model.add(Dense(2, activation = params['last_activation']))

    #compile and history
    model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = ['accuracy'])
    history = model.fit(x_train, y_train,
                        validation_data = [x_val, y_val],
                        batch_size = params['batch_size'],
                        epochs = params['epochs'],
                        verbose = 1)
                        # callbacks = [tensorboard])

    return history, model



##################### END FUNCTIONS ###############################################

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
df_test = pd.read_csv('csvs/test/b14_reset.csv')
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


################################################### Hyperparameter tuning with talos #############################################

########### Parameter Optimization ################
# p = {
#   'first_neuron': (60, 2500, 20),
#   'hidden_neurons': (50, 4000, 20),
#   'hidden_layers': (10, 5000, 20),
#   'batch_size': (10, 32, 4),
#   'optimizer': ['adam','RMSprop'],
#   'loss':['binary_crossentropy','sparse_categorical_crossentropy'],
#   'epochs': (10, 150, 5),
#   'dropout':[0.0, 0.25, 0.5],
#   'activation': ['relu','elu'],
#   'last_activation':['sigmoid']
#   }

# # quick test param
# p = {
# 'first_neuron': [10, 20],
# 'hidden_neurons': [10, 50],
# 'hidden_layers': [1, 5],
# 'batch_size': [12, 128],
# 'optimizer': ['adam'],
# 'loss':['binary_crossentropy','sparse_categorical_crossentropy'],
# 'epochs': [10, 15],
# 'dropout':[0.0,0.5],
# 'activation': ['relu'],
# 'last_activation':['sigmoid']
# }


p = {
'first_neuron': (20, 80, 4),
'hidden_neurons': (7, 497, 7),
'hidden_layers': [5, 15, 35],
'batch_size': [10, 32],
'optimizer': ['adam','RMSprop'],
'loss':['binary_crossentropy','sparse_categorical_crossentropy'],
'epochs': [1, 5, 10, 24],
'dropout':[0.0, 0.25, 0.5],
'activation': ['relu','elu'],
'last_activation':['sigmoid']
}

######### Running with talos scan ########################
t = ta.Scan(x = df_train,
         y = df_y_train,
         x_val = df_test,
         y_val = df_y_test,
         model = binary_model,
         params = p,
         clear_session=True,
         fraction_limit= 0.2,
         reduction_method='forest',
         reduction_interval=50,
         reduction_window=30,
         reduction_threshold=0.2,
         reduction_metric='mae',
         minimize_loss=True,
         experiment_name = TRAINED_MODEL_DIR+'tuned_model')


Deploy(t, 'trained_model_Data_1_val_accuracy_{}'.format(TIMESTR), metric='val_accuracy')
Deploy(t, 'trained_model_Data_1_val_loss_{}'.format(TIMESTR), metric='val_loss', asc=True)

shutil.move('trained_model_Data_1_val_accuracy_{}.zip'.format(TIMESTR),
                TRAINED_MODEL_DIR+'trained_model_Data_1_val_accuracy_{}'.format(TIMESTR))
shutil.move('trained_model_Data_1_val_loss_{}.zip'.format(TIMESTR),
                TRAINED_MODEL_DIR+'trained_model_Data_1_val_loss_{}'.format(TIMESTR))





########################################## End of Talos test #################################################
