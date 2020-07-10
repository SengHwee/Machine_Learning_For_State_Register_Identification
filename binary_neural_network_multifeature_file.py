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

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from multiprocessing import pool


import talos as ta
# from talos.model.layers import hidden_layers
from talos.model.early_stopper import early_stopper
from talos import Scan
from talos import Deploy

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
def classification_model(optimizer='adam', loss='binary_crossentropy'):
    model = Sequential()
    model.add(Dense(124, activation='relu'))  # , kernel_initializer='random_uniform'))
    model.add(Dense(302, activation='relu'))  # , kernel_initializer='random_uniform'))
    model.add(Dense(302, activation='relu'))  # , kernel_initializer='random_uniform'))
    model.add(Dense(302, activation='relu'))  # , kernel_initializer='random_uniform'))
    model.add(Dense(302, activation='relu'))  # , kernel_initializer='random_uniform'))
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
        # print("Permuting feature: {:d}, Run Number: {:d}".format(features+1, s+1))

        x_train_s[features] = np.random.permutation(x_train_s[features])  #permute feature row
        # print(x_train_s[features])  #check to ensure data is permuted
        # print(x_train.T[features])

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
        # for s in range(runs):
        #     keras_est = keras_model()
        #     x_train_s = np.copy(x_train.T)  #copy of training data to permute, permutes within array - to always start with original training data
        #     print("Permuting feature: {:d}, Run Number: {:d}".format(features+1, s+1))

        #     x_train_s[features] = np.random.permutation(x_train_s[features])  #permute feature row
        #     print(x_train_s[features])  #check to ensure data is permuted
        #     print(x_train.T[features])

        #     keras_est.fit(x_train_s.T, y_train, shuffle=False, epochs=epochs, batch_size=batch_size)  #training model with permuted data

        #     test_loss, test_acc = keras_est.evaluate(x_test, y_test, verbose=1)
        #     print (test_loss, test_acc)
        #     shuffled_acc.append(test_acc)  #holder for accuracy of runs
        #     k.clear_session()
        shuffled_acc = Parallel(n_jobs=n_jobs)(delayed(permute_columns)(s)for s in range(runs))
        all_permuted_acc_list.append(shuffled_acc)
        shuffled_feature_mean = np.array(shuffled_acc).mean()  #getting mean of all runs of the feature
        shuffled_feature_mean_list = np.append(shuffled_feature_mean_list, shuffled_feature_mean) #store in mean list for comparison and filtering
        shuffled_acc = list(np.array([]))

    all_permuted_acc_list = pd.DataFrame(all_permuted_acc_list)
    all_permuted_acc_list.to_csv(RUNDIR+"all_permuted_acc_list.csv")


    # for o in range(runs):
    #     keras_est = keras_model()
    #     keras_est.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    #     test_loss, test_acc = keras_est.evaluate(x_test, y_test)
    #     unshuffled_acc.append(test_acc)  #holder for accuracy of runs

    unshuffled_acc = Parallel(n_jobs=n_jobs)(delayed(original_data)(o) for o in range(runs))

    unshuffled_acc_pd = pd.DataFrame(unshuffled_acc)
    unshuffled_acc_pd.to_csv(RUNDIR+"unshuffled_acc_list.csv")
    unshuffled_feature_mean = np.array(unshuffled_acc).mean()
    unshuffled_feature_mean_list = np.append(unshuffled_feature_mean_list, unshuffled_feature_mean)

    #combine all mean data for comparision and filtering
    acc_list = [shuffled_feature_mean_list, unshuffled_feature_mean_list]
    acc_list_pd = pd.DataFrame(acc_list).transpose()
    acc_list_pd.rename(columns={acc_list_pd.columns[0]: "shuffled_feature_mean_accuracy", acc_list_pd.columns[1]: "unshuffled_feature_mean_accuracy"}, inplace=True)
    # print(acc_list_pd)
    acc_list_pd.to_csv(RUNDIR+"Mean_acc.csv", index=False) #save mean list

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
    feature_names_permuted_columns.to_csv(RUNDIR+"feature_names_permuted_columns.csv", index=False)
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
    feature_names_SFS.to_csv(RUNDIR+"feature_names_SFS.csv", index=False)
    k.clear_session()

    # # training model with chosen features
    # keras_est.fit(x_train_sfs, y_train)
    # y_pred = keras_est.predict(x_test_sfs)

    # # evaluating model with accuracy and false positive index
    # correct = 0
    # index_wrong=[]
    # false_positive=[]
    # y_test = y_test.flatten()
    # y_pred = y_pred.flatten()

    # # for i in range(len(y_pred)):
    # #   if y_test[i] == y_pred[i]:
    # #       correct += 1
    # #   else:
    # #       index_wrong.append(i)
    # #       if y_test[i] == 0:
    # #           false_positive.append(i)

    # for i in range(len(y_pred)):
    #   if y_test[i] != y_pred[i]:
    #       index_wrong.append(i)
    #       if y_test[i] == 0:
    #         false_positive.append(i)

    # # checking model accuracy:
    # percent_correct= accuracy_score(y_test, y_pred)
    # accuracy_result = pd.DataFrame.from_dict(sk_keras_est.get_metric_dict()).T
    # accuracy_result.to_csv(DATADIR+"accuracy_result.csv", index=False)

    # print('Selected features:', sk_keras_est.k_feature_idx_)
    # #percent_correct = (correct/len(df_y_test))
    # print("Model accurary is: {:.2f}%".format(percent_correct*100))
    # print("Wrong prediction index: ", index_wrong)
    # print("Index with False Positive: ", false_positive)


    return x_train_sfs, x_test_sfs #return original dataframe if none is dropped



########### Creation of Model Function  for Talos ##############
def binary_model (x_train, y_train, x_val, y_val, params):
    # print(params)
    ########### Tracking performance on tensorboard ####################
    tensorboard = TensorBoard(log_dir = TRAINED_MODEL_DIR+'logs/{}'.format(params))

    model = Sequential()

    #initial layer
    model.add(Dense(params['first_neuron'], input_dim = x_train.shape[1], activation = params['activation']))
    model.add(Dropout(params['dropout']))

    #hidden layers
    for i in range (params['hidden_layers']):
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
                        verbose = 1,
                        callbacks = [tensorboard])

    return history, model


########## creation of directory##########
TIMESTR = time.strftime("%Y%m%d-%H%M%S")
TRAINDIR = 'csvs/train/'
TESTDIR = 'csvs/test/'
CSVDIR = 'csvs/'
DATADIR = 'data/runs_{}/'.format(TIMESTR)

if not os.path.exists(DATADIR):
    os.makedirs(DATADIR)

#getting all the data files with .csv
data_files = [train_data for train_data in os.listdir(CSVDIR) if '.csv' in train_data]

# To loop through all feature files, using each one as test files and the rest as train
for test_file in data_files:
    #creation of feature file test folder in Data Directory
    RUNDIR = DATADIR+'{}_test_file/'.format(test_file.replace('.csv', '_'))
    TRAINED_MODEL_DIR = RUNDIR+'model/'
    if not os.path.exists(TRAINED_MODEL_DIR):
        os.makedirs(TRAINED_MODEL_DIR)

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
    df_test = pd.read_csv('csvs/test/{}'.format(test_file))
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
    # df_train = quasi_constant_filter.fit_transform(df_train) #quasi filter not used as accuracy reduced significantly
    # df_test = quasi_constant_filter.transform(df_test)        #removed too many features

    # def get_correlation(data, threshold):
    #   corr_col = set()
    #   corr_matrix = data.corr()
    #   for i in range(len(corr_matrix.columns)):
    #       for j in range(i):
    #           if abs(corr_matrix.iloc[i, j]) > threshold:
    #               colname = corr_matrix.columns[i]
    #               corr_col.add(colname)
    #   return corr_col

    # df_train = pd.DataFrame(df_train)
    # df_test = pd.DataFrame(df_test)
    # df_train = df_train.T
    # df_test = df_test.T
    # duplicated_features = df_train.duplicated()
    # features_to_keep = [not index for index in duplicated_features]
    # df_train = df_train[features_to_keep].T
    # df_test = df_test[features_to_keep].T

    # corr_features = get_correlation(df_train, 0.85)
    # df_train = df_train.drop(labels = corr_features, axis = 1)
    # df_test = df_test.drop(labels = corr_features, axis = 1)

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
    sk_model = KerasClassifier(build_fn=classification_model,
                                   epochs=15, batch_size=12)

    # feature permutation by coloumn permutation
    df_train, df_test = fs_column_permutation(df_train, df_y_train, df_test, df_y_test, classification_model, epochs=15, batch_size=12,
                                            runs=2, variance=0.01, n_jobs=-1)

    # feature permutation by step selection
    df_train, df_test = step_feature_selection(sk_model, df_train, df_y_train, df_test, df_y_test,
                            1, df_train.shape[1], scoring='accuracy', cv=0, n_jobs=-1)

    ####################################### Step feature selection (moved to function)- ignore #########################################
    # # feature selection step forward/backward:
    # keras_clf = SFS(binary_model, k_features=(1, 12), forward=True,
    #                 floating=False, verbose=1, scoring='accuracy', cv=0, n_jobs=-1)

    # # feature selection exhaustive feature selection:
    # keras_clf = EFS(binary_model, min_features=6, max_features=10,
    #                 scoring='accuracy', cv=0, n_jobs=-1)


    # # transforming data to only contain chosen features:
    # keras_clf = keras_clf.fit(df_train, df_y_train)

    # df_train_sfs = keras_clf.transform(df_train)
    # df_test_sfs = keras_clf.transform(df_test)

    # # training model with chosen features
    # binary_model.fit(df_train_sfs, df_y_train)
    # y_pred = binary_model.predict(df_test_sfs)

    # # evaluating model with accuracy and false positive index
    # correct = 0
    # index_wrong=[]
    # false_positive=[]
    # df_y_test = df_y_test.flatten()
    # y_pred = y_pred.flatten()

    # for i in range(len(y_pred)):
    #   if df_y_test[i] == y_pred[i]:
    #       correct += 1
    #   else:
    #       index_wrong.append(i)
    #       if df_y_test[i] == 0:
    #           false_positive.append(i)

    # # # checking model accuracy:
    # # accuracy_result = pd.DataFrame.from_dict(keras_clf.get_metric_dict()).T
    # # accuracy_result.to_csv("accuracy_result", index=False)
    # # fig1 = plot_sfs(keras_clf.get_metric_dict(), kind='std_dev')
    # # plt.ylim([0.8, 1])
    # # plt.title('Sequential Forward Selection (w. StdDev)')
    # # plt.grid()
    # # plt.show()

    # print('Selected features:', keras_clf.k_feature_idx_)
    # percent_correct = (correct/len(df_y_test))
    # print("Model accurary is: {:.2f}%".format(percent_correct*100))
    # print("Wrong prediction index: ", index_wrong)
    # print("Index with False Positive: ", false_positive)


    ############################ Feature Selection: Backward + model training ##########################################

    # # Creation of Model:
    # def classification_model(optimizer='adam', loss='binary_crossentropy', *,input_shape):
    #     model = Sequential()
    #     model.add(Dense(124, activation='relu', input_shape=(input_shape,)))  # , kernel_initializer='random_uniform'))
    #     model.add(Dense(302, activation='relu'))  # , kernel_initializer='random_uniform'))
    #     model.add(Dense(302, activation='relu'))  # , kernel_initializer='random_uniform'))
    #     model.add(Dense(302, activation='relu'))  # , kernel_initializer='random_uniform'))
    #     model.add(Dense(302, activation='relu'))  # , kernel_initializer='random_uniform'))
    #     model.add(Dense(1, activation='sigmoid'))

    #     model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    #     return model

    # df_train = pd.DataFrame(df_train)
    # df_test = pd.DataFrame(df_test)
    # print(df_train)
    # print(df_test)


    # # step backward feature selection + training of model:
    # accuracy_list=[]
    # for features in range(len(df_train.columns)):
    #   print(features)
    #   if features > 0:
    #       df_train = pd.DataFrame(df_train)
    #       df_train = df_train.drop(df_train.columns[0], axis=1)
    #       df_test = pd.DataFrame(df_test)
    #       df_test = df_test.drop(df_test.columns[0], axis=1)
    #       print(df_train.shape)
    #       print(df_test.shape)
    #       print(df_train)
    #       print(df_test)
    #   binary_model = classification_model(input_shape=len(df_train.columns))
    #   binary_model.fit(df_train.to_numpy(),df_y_train.to_numpy(), batch_size=12, epochs=10, verbose=1)
    #   accuracy_list.append(binary_model.evaluate(df_test.to_numpy(), df_y_test.to_numpy()))
    # print(accuracy_list)

    ############################################### END ################################################################

    ############## training model ######################
    # print(type(df_train), type(df_y_train))

    # model.fit(df_train, df_y_train.to_numpy(), batch_size = 12, epochs = 10, verbose = 1)
    # df_y_pred = model.predict_classes(df_test)
    # print(df_y_pred, df_y_test)
    # model.evaluate(df_test, df_y_test)

    ################################################### Hyperparameter tuning with talos #############################################

    ########### Parameter Optimization ################
    # p = {
    #   'first_neuron': (10, 80, 5),
    #   'hidden_neurons': (10, 60, 5),
    #   'hidden_layers': (1, 20, 5),
    #   'batch_size': [12, 24, 32],
    #   'optimizer': ['adam','RMSprop'],
    #   'loss':['binary_crossentropy','sparse_categorical_crossentropy'],
    #   'epochs': (10, 30, 5),
    #   'dropout':[0.0, 0.25, 0.5],
    #   'activation': ['relu', 'elu'],
    #   'last_activation':['sigmoid']
    #   }


    #quick test param
    p = {
    'first_neuron': [10],
    'hidden_neurons': [10],
    'hidden_layers': [5],
    'batch_size': [12],
    'optimizer': ['adam'],
    'loss':['binary_crossentropy'],
    'epochs': [10],
    'dropout':[0.0],
    'activation': ['relu'],
    'last_activation':['sigmoid']
    }


    ########## Running with talos scan ########################
    t = ta.Scan(x = df_train,
             y = df_y_train,
             x_val = df_test,
             y_val = df_y_test,
             model = binary_model,
             params = p,
             experiment_name = RUNDIR+'100_Runs_Feature_Selection_Data_1')


    Deploy(t, 'trained_model_Data_1_val_accuracy_{}_{}'.format(test_file, TIMESTR), metric='val_accuracy')
    Deploy(t, 'trained_model_Data_1_val_loss_{}_{}'.format(test_file, TIMESTR), metric='val_loss', asc=True)
    shutil.move('trained_model_Data_1_val_accuracy_{}_{}.zip'.format(test_file, TIMESTR),
                TRAINED_MODEL_DIR+'trained_model_Data_1_val_accuracy_{}_{}'.format(test_file, TIMESTR))
    shutil.move('trained_model_Data_1_val_loss_{}_{}.zip'.format(test_file, TIMESTR),
                TRAINED_MODEL_DIR+'trained_model_Data_1_val_loss_{}_{}'.format(test_file, TIMESTR))


########################################## End of Talos test #################################################
