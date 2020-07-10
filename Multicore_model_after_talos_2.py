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
############# transpose, reorder and drop ####################
def transpose_data(df):
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df.reset_index()
    df = df.drop(df.index[0])
    df = df.reset_index()
    df = df.drop(["level_0", "index"],
                 axis=1)
    return df

############ Feature Selection +  model training (mlxtend framework) ##########

# Creation of Model:
def classification_model(x_train, hidden_layers, first_neuron, hidden_neurons, optimizer='adam', loss='binary_crossentropy'):
    model = Sequential()
    model.add(Dense(first_neuron, activation='elu'))  

    for _ in range(hidden_layers):
        model.add(Dense(hidden_neurons, input_shape= (x_train.shape[1],), activation='elu'))
        model.add(Dropout(0.25))
    
    if loss == 'binary_crossentropy':
        model.add(Dense(1, activation='sigmoid'))
    else:
        model.add(Dense(2, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


def run_data(kerasmodel, x_train, y_train, x_test, y_test, *,epochs=12, batch_size=10, r):
    keras_model=kerasmodel(x_train, hidden_layers=15, first_neuron=20, hidden_neurons=217, optimizer='RMSprop', loss='binary_crossentropy')
    keras_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    y_pred = keras_model.predict_classes(x_test)

    true_positive, isStateReg, isStateRegAcc, model_acc = checker.true_positive_checker(y_test, y_pred)

    # print("Predicted State Register Accuracy= {}".format(isStateRegAcc))
    # print("Accuracy = {}".format(model_acc))

    k.clear_session()
    return model_acc, isStateRegAcc





name = "fastRELIC"
########## creation of directory##########
TIMESTR = time.strftime("%Y%m%d-%H%M%S")
TRAINDIR = './csvs/train_{}/'.format(name)
TESTDIR = './csvs/test_{}/'.format(name)
CSVDIR = './csvs/{}/'.format(name)
DATADIR = 'data/runs_{}_{}/'.format(TIMESTR, name)
AVERAGE_RESULTS = DATADIR + 'average_results/'


if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)
test_file_names = {}

#getting all the data files with .csv
data_files = [train_data for train_data in os.listdir(CSVDIR) if '.csv' in train_data]

# To loop through all feature files, using each one as test files and the rest as train
for test_file in data_files:
    average_acc_100_list = []
    isStateReg_acc_100_list = []
    for average_runs in range(5):
        #creation of feature file test folder in Data Directory
        RUNDIR = DATADIR+'{}_test_file/'.format(test_file.replace('.csv', '_'))
        TRAINED_MODEL_DIR = RUNDIR+'model/'
        if not os.path.exists(TRAINED_MODEL_DIR):
            os.makedirs(TRAINED_MODEL_DIR)
        if not os.path.exists(AVERAGE_RESULTS):
            os.makedirs(AVERAGE_RESULTS)

        #copying the list of training file to drop the test file
        training_files = data_files.copy()
        # Deleting previous train and test folders to copy new train and test files
        if os.path.exists(TRAINDIR) and os.path.exists(TESTDIR):
            shutil.rmtree(TRAINDIR)
            os.makedirs(TRAINDIR)
            shutil.rmtree(TESTDIR)
            os.makedirs(TESTDIR)
        else:
            os.makedirs(TRAINDIR)
            os.makedirs(TESTDIR)

        #copy test file to test folder
        shutil.copyfile(CSVDIR+test_file, TESTDIR+test_file)
        #removing test file from list
        training_files.remove(test_file)
        #copying training files to train folder
        for train_files in training_files:
            shutil.copyfile(CSVDIR+train_files, TRAINDIR+train_files)

        #combining all training files into 1 training file
        data_array = []
        for train_data in os.listdir(TRAINDIR):
            df = pd.read_csv(TRAINDIR + train_data)
            df = transpose_data(df)
            data_array.append(df)

        df_train = pd.concat(data_array)
        df_train.to_csv(RUNDIR+"combine_training_data.csv", index=False)
        df_y_train = df_train.pop('xx_state_ff')
        df_train = df_train.drop(['xx_good_SN'], axis=1)
        df_train = df_train.drop(['has_feedback_path'], axis=1)
        df_train = df_train.drop(['load_centrality'], axis=1)
        # print(df_y_train)
        # print(df_train)

        ################ load testing data ##################
        target_name = 'xx_state_ff'
        df_test = pd.read_csv(TESTDIR+'{}'.format(test_file))
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
        # quasi_constant_filter = VarianceThreshold(threshold=0.01)  # remove columns with 1% difference in values
        constant_filter_df_train = constant_filter.fit_transform(df_train)
        constant_filter_df_test = constant_filter.transform(df_test)


        feature_names = list(df_train[df_train.columns[constant_filter.get_support(indices=True)]].columns)
        df_train = constant_filter_df_train
        df_test = constant_filter_df_test
        feature_names_constant_filter = pd.DataFrame(feature_names)
        feature_names_constant_filter.to_csv(RUNDIR+"feature_names_constant_filter.csv", index=False)

        ################ scaling and converting of data ###################
        scalar = StandardScaler()

        df_train = scalar.fit_transform(df_train)
        df_test = scalar.transform(df_test)

        df_y_train = df_y_train.to_numpy().astype(int)
        df_y_test = df_y_test.to_numpy().astype(int)


        accuracy_array, isStateRegAcc_array = zip(*Parallel(n_jobs=40)(delayed(run_data)(classification_model, df_train, df_y_train, df_test, df_y_test, r=r) for r in range(100)))    

        accuracy_list = [accuracy_array, [np.average(accuracy_array)], isStateRegAcc_array, [np.average(isStateRegAcc_array)]]
        accuracy_list = pd.DataFrame(accuracy_list).transpose()
        accuracy_list.rename(columns={accuracy_list.columns[0]: "{}_accuracy".format(name),accuracy_list.columns[1]: "{}_average".format(name),
                            accuracy_list.columns[2]: "State_Register_accuracy",accuracy_list.columns[3]: "State_Register_average"}, inplace=True)
        accuracy_list.to_csv(AVERAGE_RESULTS + "{}_{}_score_run_{}.csv".format(name, test_file, average_runs+1), index= False)
        average_acc_100_list.append(np.average(accuracy_array))
        isStateReg_acc_100_list.append(np.average(isStateRegAcc_array))
    all_score = [np.average(average_acc_100_list), np.average(isStateReg_acc_100_list)]
    test_file_names[test_file] = all_score
test_file_names = pd.DataFrame.from_dict(test_file_names, orient='index', columns=['5x100: Runs Accuracy', '5x100 Runs: State Register Accuracy'])
test_file_names.to_csv(AVERAGE_RESULTS + "average_scores.csv")
