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
    shuffled_feature_mean_list = []
    unshuffled_feature_mean_list = []
    acc_list = []
    all_permuted_acc_list = []

    def permute_columns(s):
        keras_est = keras_model()
        x_train_s = np.copy(x_train.T)  #copy of training data to permute, permutes within array - to always start with original training data
        # print("Permuting feature: {:d}, Run Number: {:d}".format(features+1, s+1))

        x_train_s[features] = np.random.permutation(x_train_s[features])  #permute feature row
        # print(x_train_s[features])  #check to ensure data is permuted
        # print(x_train.T[features])

        keras_est.fit(x_train_s.T, y_train, shuffle=False, epochs=epochs, batch_size=batch_size)  #training model with permuted data
        # test_loss, test_acc = keras_est.evaluate(x_test, y_test, verbose=1)
        y_predP = keras_est.predict_classes(x_test)
        _, _, test_acc, _ = checker.true_positive_checker(y_test, y_predP)

        k.clear_session()
        return test_acc

    def original_data(o):
        keras_est = keras_model()
        keras_est.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        # test_loss, test_acc = keras_est.evaluate(x_test, y_test)
        y_predO = keras_est.predict_classes(x_test)
        _, _, test_acc, _ = checker.true_positive_checker(y_test, y_predO)
        k.clear_session()
        return test_acc



    for features in range(len(x_train.T)):  #len return number of rows, transpose to get total number of features

        shuffled_acc = Parallel(n_jobs=n_jobs)(delayed(permute_columns)(s)for s in range(runs))
        all_permuted_acc_list.append(shuffled_acc)
        shuffled_feature_mean = np.array(shuffled_acc).mean()  #getting mean of all runs of the feature
        shuffled_feature_mean_list = np.append(shuffled_feature_mean_list, shuffled_feature_mean) #store in mean list for comparison and filtering
        shuffled_acc = list(np.array([]))

    all_permuted_acc_list = pd.DataFrame(all_permuted_acc_list)
    all_permuted_acc_list.to_csv(RUNDIR+"all_permuted_acc_list_{}.csv".format(average_runs))


    unshuffled_acc = Parallel(n_jobs=n_jobs)(delayed(original_data)(o) for o in range(runs))

    unshuffled_acc_pd = pd.DataFrame(unshuffled_acc)
    unshuffled_acc_pd.to_csv(RUNDIR+"unshuffled_acc_list_{}.csv".format(average_runs))
    unshuffled_feature_mean = np.array(unshuffled_acc).mean()
    unshuffled_feature_mean_list = np.append(unshuffled_feature_mean_list, unshuffled_feature_mean)

    #combine all mean data for comparision and filtering
    acc_list = [shuffled_feature_mean_list, unshuffled_feature_mean_list]
    acc_list_pd = pd.DataFrame(acc_list).transpose()
    acc_list_pd.rename(columns={acc_list_pd.columns[0]: "shuffled_feature_mean_accuracy", acc_list_pd.columns[1]: "unshuffled_feature_mean_accuracy"}, inplace=True)
    # print(acc_list_pd)
    acc_list_pd.to_csv(RUNDIR+"StateReg_Mean_acc_{}.csv".format(average_runs), index=False) #save mean list

    #slection of features to drop based on equal values and variance threshold
    global feature_names
    feature_index_to_drop = []
    feature_index_to_drop = [i for i in range(len(acc_list_pd))
                            if (acc_list_pd.iloc[i, 0]==acc_list_pd.iloc[0, 1]) or (acc_list_pd.iloc[i, 0]>acc_list_pd.iloc[0, 1]) or (abs(acc_list_pd.iloc[i, 0] - acc_list_pd.iloc[0, 1])/acc_list_pd.iloc[0, 1]<variance)]
    # print(feature_index_to_drop)
    # print(len(feature_index_to_drop))
    # print(feature_names)

    #dropping of features if any
    if len(feature_index_to_drop) is not 0:
        x_train = pd.DataFrame(x_train)
        x_test = pd.DataFrame(x_test)
        # print(x_train)
        # print(x_test)
        for i in list(reversed(feature_index_to_drop)):  #have to reverse order as dropping earlier feature will result in missing columns
            x_train = x_train.drop(x_train.columns[i], axis=1)  #drop column(s) in both train and test files to ensure same dimension
            x_test = x_test.drop(x_test.columns[i], axis=1)
            del feature_names[i]
        # print(feature_names)
        # print(x_train)
        # print(x_test)
        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()

    feature_names_permuted_columns = pd.DataFrame(feature_names)
    feature_names_permuted_columns.to_csv(RUNDIR+"feature_names_permuted_columns_{}.csv".format(average_runs), index=False)
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
    feature_names_SFS.to_csv(RUNDIR+"feature_names_SFS_{}.csv".format(average_runs), index=False)
    k.clear_session()


    return x_train_sfs, x_test_sfs #return original dataframe if none is dropped



# ########### Creation of Model Function  for Talos ##############
# def binary_model (x_train, y_train, x_val, y_val, params):
#     # print(params)
#     ########### Tracking performance on tensorboard ####################
#     tensorboard = TensorBoard(log_dir = TRAINED_MODEL_DIR+'logs/{}'.format(params))
#
#     model = Sequential()
#
#     #initial layer
#     model.add(Dense(params['first_neuron'], input_dim = x_train.shape[1], activation = params['activation']))
#     model.add(Dropout(params['dropout']))
#
#     #hidden layers
#     for i in range (params['hidden_layers']):
#         # print(f"adding layer {i+1}")
#         model.add(Dense(params['hidden_neurons'], activation = params['activation']))
#         model.add(Dropout(params['dropout']))
#
#     #final layer
#     if params['loss'] == 'binary_crossentropy':
#         model.add(Dense(1, activation = params['last_activation']))
#     else:
#         model.add(Dense(2, activation = params['last_activation']))
#
#     #compile and history
#     model.compile(loss = params['loss'], optimizer = params['optimizer'], metrics = ['accuracy'])
#     history = model.fit(x_train, y_train,
#                         validation_data = [x_val, y_val],
#                         batch_size = params['batch_size'],
#                         epochs = params['epochs'],
#                         verbose = 1,
#                         callbacks = [tensorboard])
#
#     return history, model

def run_data(kerasmodel, x_train, y_train, x_test, y_test, *,epochs=22, batch_size=12, r):
    keras_model=kerasmodel()
    keras_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose = 1)
    y_pred = keras_model.predict_classes(x_test)

    true_positive, isStateReg, isStateRegAcc, model_acc = checker.true_positive_checker(y_test, y_pred)
    false_positive = checker.false_positive_checker(y_test, y_pred)
    keras_model.evaluate(x_test, y_test)

    # print("Index with True Positive: ", true_positive)
    # print("Index with False Positive: ", false_positive)
    # print("Predicted State Register Index= {}".format(isStateReg))
    # print("Predicted State Register Accuracy= {}".format(isStateRegAcc))
    # print("Accuracy = {}".format(model_acc))

    k.clear_session()
    return model_acc, isStateRegAcc


name = "euclidean"
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
        # print(df_y_train)
        # print(df_train)

        ################ load testing data ##################
        target_name = 'xx_state_ff'
        df_test = pd.read_csv(TESTDIR+'{}'.format(test_file))
        df_test = transpose_data(df_test)
        df_y_test = df_test.pop(target_name)
        df_test = df_test.drop(['xx_good_SN'], axis=1)
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
        # print(feature_names)
        # print(pd.DataFrame(df_train))
        # print(pd.DataFrame(df_test))
        # print(df_train.corr())

        ################ scaling and converting of data ###################
        scalar = StandardScaler()

        df_train = scalar.fit_transform(df_train)
        df_test = scalar.transform(df_test)

        df_y_train = df_y_train.to_numpy().astype(int)
        df_y_test = df_y_test.to_numpy().astype(int)

        # initialise and convert Keras model to scikit learn:
        # sk_model = KerasClassifier(build_fn=classification_model,
        #                                epochs=22, batch_size=12)
        
        # print(df_y_train)
        # print(df_y_train)
        # feature permutation by coloumn permutation
        df_train, df_test = fs_column_permutation(df_train, df_y_train, df_test, df_y_test, classification_model, epochs=22, batch_size=12,
                                                runs=100, variance=0.01, n_jobs=-1)
        
        # if len(feature_names) != 0:
            # feature permutation by step selection
        # df_train, df_test = step_feature_selection(sk_model, df_train, df_y_train, df_test, df_y_test,
        #                         1, df_train.shape[1], scoring='accuracy', cv=0, n_jobs=20)



        ################################################### Hyperparameter tuning with talos #############################################

    #     ########### Parameter Optimization ################
    #     p = {
    #       'first_neuron': (10, 80, 5),
    #       'hidden_neurons': (10, 60, 5),
    #       'hidden_layers': (1, 20, 5),
    #       'batch_size': [12, 24, 32],
    #       'optimizer': ['adam','RMSprop'],
    #       'loss':['binary_crossentropy','sparse_categorical_crossentropy'],
    #       'epochs': (10, 30, 5),
    #       'dropout':[0.0, 0.25, 0.5],
    #       'activation': ['relu', 'elu'],
    #       'last_activation':['sigmoid']
    #       }
    #
    #
    #     # #quick test param
    #     # p = {
    #     # 'first_neuron': [10],
    #     # 'hidden_neurons': [10],
    #     # 'hidden_layers': [5],
    #     # 'batch_size': [12],
    #     # 'optimizer': ['adam'],
    #     # 'loss':['binary_crossentropy'],
    #     # 'epochs': [10],
    #     # 'dropout':[0.0],
    #     # 'activation': ['relu'],
    #     # 'last_activation':['sigmoid']
    #     # }
    #
    #
    #     ########## Running with talos scan ########################
    #     t = ta.Scan(x = df_train,
    #              y = df_y_train,
    #              x_val = df_test,
    #              y_val = df_y_test,
    #              model = binary_model,
    #              params = p,
    #              experiment_name = RUNDIR+'100_Runs_Feature_Selection_Data_1')
    #
    #
    #     Deploy(t, 'trained_model_Data_1_val_accuracy_{}_{}'.format(test_file, TIMESTR), metric='val_accuracy')
    #     Deploy(t, 'trained_model_Data_1_val_loss_{}_{}'.format(test_file, TIMESTR), metric='val_loss', asc=True)
    #     shutil.move('trained_model_Data_1_val_accuracy_{}_{}.zip'.format(test_file, TIMESTR),
    #                 TRAINED_MODEL_DIR+'trained_model_Data_1_val_accuracy_{}_{}'.format(test_file, TIMESTR))
    #     shutil.move('trained_model_Data_1_val_loss_{}_{}.zip'.format(test_file, TIMESTR),
    #                 TRAINED_MODEL_DIR+'trained_model_Data_1_val_loss_{}_{}'.format(test_file, TIMESTR))
    #
    #
    # ########################################## End of Talos test #################################################
#         accuracy_array = []
#         isStateRegAcc_array=[]
#         for run in range(100):
#             k.clear_session()
#             keras_model = classification_model()
#             keras_model.fit(df_train, df_y_train, batch_size = 12, epochs = 22, verbose = 1)
#             df_y_pred = keras_model.predict_classes(df_test)

#             true_positive, isStateReg, isStateRegAcc, model_acc = checker.true_positive_checker(df_y_test, df_y_pred)
#             false_positive = checker.false_positive_checker(df_y_test, df_y_pred)
#             keras_model.evaluate(df_test, df_y_test)

#             # print("Index with True Positive: ", true_positive)
#             # print("Index with False Positive: ", false_positive)
#             # print("Predicted State Register Index= {}".format(isStateReg))
#             # print("Predicted State Register Accuracy= {}".format(isStateRegAcc))
#             # print("Accuracy = {}".format(model_acc))
#             accuracy_array.append(model_acc)
#             isStateRegAcc_array.append(isStateRegAcc)

#         # accuracy_array, isStateRegAcc_array = Parallel(n_jobs=-1)(delayed(run_data)(classification_model, df_train, df_y_train, df_test, df_y_test, r=r) for r in range(100))
                

#         accuracy_list = [accuracy_array, [np.average(accuracy_array)], isStateRegAcc_array, [np.average(isStateRegAcc_array)]]
#         accuracy_list = pd.DataFrame(accuracy_list).transpose()
#         accuracy_list.rename(columns={accuracy_list.columns[0]: "{}_accuracy".format(name),accuracy_list.columns[1]: "{}_average".format(name),
#                             accuracy_list.columns[2]: "State_Register_accuracy",accuracy_list.columns[3]: "State_Register_average"}, inplace=True)
#         accuracy_list.to_csv(AVERAGE_RESULTS + "{}_{}_score_run_{}.csv".format(name, test_file, average_runs+1), index= False)
#         average_acc_100_list.append(np.average(accuracy_array))
#         isStateReg_acc_100_list.append(np.average(isStateRegAcc_array))
#     all_score = [np.average(average_acc_100_list), np.average(isStateReg_acc_100_list)]
#     test_file_names[test_file] = all_score
# test_file_names = pd.DataFrame.from_dict(test_file_names, orient='index', columns=['5x100: Runs Accuracy', '5x100 Runs: State Register Accuracy'])
# test_file_names.to_csv(AVERAGE_RESULTS + "average_scores.csv")
        
