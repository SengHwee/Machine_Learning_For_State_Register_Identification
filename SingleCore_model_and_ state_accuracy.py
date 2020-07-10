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
def classification_model(optimizer='RMSprop', loss='binary_crossentropy'):
    model = Sequential()
    model.add(Dense(66, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model


# feature selection by permuting columns
def fs_column_permutation(x_train, y_train, x_test, y_test, keras_model, epochs, batch_size, runs, variance, n_jobs=-1):
    #y_test = y_test.flatten()
    # shuffled_acc = []
    # unshuffled_acc = []
    shuffled_feature_mean_list = []
    unshuffled_feature_mean_list = []
    acc_list = []
    all_permuted_acc_list = []

    def permute_columns(s):
        keras_est = keras_model()
        x_train_s = np.copy(x_train.T)  #copy of training data to permute, permutes within array - to always start with original training data
        x_train_s[features] = np.random.permutation(x_train_s[features])  #permute feature row
        keras_est.fit(x_train_s.T, y_train, shuffle=False, epochs=epochs, batch_size=batch_size)  #training model with permuted data
        test_loss, test_acc = keras_est.evaluate(x_test, y_test, verbose=1)
        k.clear_session()
        return test_acc

    def original_data(o):
        keras_est = keras_model()
        keras_est.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        test_loss, test_acc = keras_est.evaluate(x_test, y_test)
        k.clear_session()
        return test_acc

    for features in range(len(x_train.T)):  #len return number of rows, transpose to get total number of features
        shuffled_acc = Parallel(n_jobs=n_jobs)(delayed(permute_columns)(s)for s in range(runs))
        all_permuted_acc_list.append(shuffled_acc)
        shuffled_feature_mean = np.array(shuffled_acc).mean()  #getting mean of all runs of the feature
        shuffled_feature_mean_list = np.append(shuffled_feature_mean_list, shuffled_feature_mean) #store in mean list for comparison and filtering
        shuffled_acc = list(np.array([]))

    all_permuted_acc_list = pd.DataFrame(all_permuted_acc_list)
    all_permuted_acc_list.to_csv(DATADIR+"all_permuted_acc_list.csv")


    unshuffled_acc = Parallel(n_jobs=n_jobs)(delayed(original_data)(o) for o in range(runs))

    unshuffled_acc_pd = pd.DataFrame(unshuffled_acc)
    unshuffled_acc_pd.to_csv(DATADIR+"unshuffled_acc_list.csv")
    unshuffled_feature_mean = np.array(unshuffled_acc).mean()
    unshuffled_feature_mean_list = np.append(unshuffled_feature_mean_list, unshuffled_feature_mean)

    #combine all mean data for comparision and filtering
    acc_list = [shuffled_feature_mean_list, unshuffled_feature_mean_list]
    acc_list_pd = pd.DataFrame(acc_list).transpose()
    acc_list_pd.rename(columns={acc_list_pd.columns[0]: "shuffled_feature_mean_accuracy", acc_list_pd.columns[1]: "unshuffled_feature_mean_accuracy"}, inplace=True)
    # print(acc_list_pd)
    acc_list_pd.to_csv(DATADIR+"Mean_acc.csv", index=False) #save mean list

    #slection of features to drop based on equal values and variance threshold
    global feature_names
    feature_index_to_drop = []
    feature_index_to_drop = [i for i in range(len(acc_list_pd))
                            if (acc_list_pd.iloc[i, 0]==acc_list_pd.iloc[0, 1]) or (acc_list_pd.iloc[i, 0]>acc_list_pd.iloc[0, 1]) or (abs(acc_list_pd.iloc[i, 0] - acc_list_pd.iloc[0, 1])/acc_list_pd.iloc[0, 1]<variance)]

    #dropping of features if any
    if len(feature_index_to_drop) is not 0:
        x_train = pd.DataFrame(x_train)
        x_test = pd.DataFrame(x_test)

        for i in list(reversed(feature_index_to_drop)):  #have to reverse order as dropping earlier feature will result in missing columns
            x_train = x_train.drop(x_train.columns[i], axis=1)  #drop column(s) in both train and test files to ensure same dimension
            x_test = x_test.drop(x_test.columns[i], axis=1)
            del feature_names[i]

        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()

    feature_names_permuted_columns = pd.DataFrame(feature_names)
    feature_names_permuted_columns.to_csv(DATADIR+"feature_names_permuted_columns.csv", index=False)
    return x_train, x_test #return original dataframe if none is dropped


def step_feature_selection(keras_est, x_train, y_train, x_test, y_test,
                           features_lower_bound, features_upper_bound, *, scoring='accuracy', cv=0, n_jobs=-1):
    # feature selection step forward/backward:
    sk_keras_est = SFS(keras_est, k_features=(features_lower_bound, features_upper_bound), forward=True,
                    floating=False, verbose=2, scoring=scoring, cv=cv, n_jobs=n_jobs)

    sk_keras_est = sk_keras_est.fit(x_train, y_train)

    # transforming data to only contain chosen features:
    x_train_sfs = sk_keras_est.transform(x_train)
    x_test_sfs = sk_keras_est.transform(x_test)

    # print(pd.DataFrame(x_train_sfs))
    # print(pd.DataFrame(x_test_sfs))

    global feature_names
    selected_features = []
    selected_features = [feature_names[i] for i in sk_keras_est.k_feature_idx_]
    feature_names = selected_features
    #print(feature_names)
    feature_names_SFS=pd.DataFrame(feature_names)
    feature_names_SFS.to_csv(DATADIR+"feature_names_SFS.csv", index=False)
    k.clear_session()

    return x_train_sfs, x_test_sfs #return original dataframe if none is dropped

name = "modified"
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

# print(df_y_train)
# print(df_train)

################ load testing data ##################
target_name = 'xx_state_ff'
df_test = pd.read_csv('csvs/test/uart.csv')
df_test = transpose_data(df_test)
df_y_test = df_test.pop(target_name)
df_test = df_test.drop(['xx_good_SN'], axis=1)

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



# # initialise and convert Keras model to scikit learn:
# sk_model = KerasClassifier(build_fn=classification_model,
#                                epochs=15, batch_size=12)
#
# # feature permutation by coloumn permutation
# df_train, df_test = fs_column_permutation(df_train, df_y_train, df_test, df_y_test, classification_model, epochs=15, batch_size=12,
#                                         runs=4, variance=0.01, n_jobs=-1)
#
# # feature permutation by step selection
# df_train, df_test = step_feature_selection(sk_model, df_train, df_y_train, df_test, df_y_test,
#                         1, df_train.shape[1], scoring='accuracy', cv=0, n_jobs=-1)
accuracy_array = []
isStateRegAcc_array=[]
for run in range(3):

    keras_model = classification_model()
    keras_model.fit(df_train, df_y_train, batch_size = 12, epochs = 22, verbose = 1)
    df_y_pred = keras_model.predict_classes(df_test)

    true_positive, isStateReg, isStateRegAcc, model_acc = checker.true_positive_checker(df_y_test, df_y_pred)
    false_positive = checker.false_positive_checker(df_y_test, df_y_pred)
    keras_model.evaluate(df_test, df_y_test)

    print("Index with True Positive: ", true_positive)
    print("Index with False Positive: ", false_positive)
    print("Predicted State Register Index= {}".format(isStateReg))
    print("Predicted State Register Accuracy= {}".format(isStateRegAcc))
    print("Accuracy = {}".format(model_acc))
    accuracy_array.append(model_acc)
    isStateRegAcc_array.append(isStateRegAcc)
    k.clear_session()

accuracy_list = [accuracy_array, [np.average(accuracy_array)], isStateRegAcc_array, [np.average(isStateRegAcc_array)]]
accuracy_list = pd.DataFrame(accuracy_list).transpose()

accuracy_list.rename(columns={accuracy_list.columns[0]: "{}_accuracy".format(name),accuracy_list.columns[1]: "{}_average".format(name),
                    accuracy_list.columns[2]: "State_Register_accuracy",accuracy_list.columns[3]: "State_Register_average"}, inplace=True)
accuracy_list.to_csv("./results/{}_score.csv".format(name), index= False)
