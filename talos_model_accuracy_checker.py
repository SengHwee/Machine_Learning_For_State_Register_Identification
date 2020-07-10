import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
from tensorflow import keras
import talos as ta
from talos.commands.restore import Restore
from talos import Predict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def transpose_data(df):
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df.reset_index()
    df = df.drop(df.index[0])
    df = df.reset_index()
    df = df.drop(["level_0", "index"],
                 axis=1)
    return df

def false_positive_checker(y_test, y_pred):

    correct = 0
    index_wrong=[]
    false_positive=[]
    # y_test = y_test.to_numpy()
    y_pred = y_pred.flatten()

    for i in range(len(y_pred)):
      if y_test[i] == y_pred[i]:
          correct += 1
      else:
          index_wrong.append(i)
          if y_test[i] == 0:
              false_positive.append(i)
    return false_positive

def true_positive_checker(y_test, y_pred):
    correct = 0
    state_register_index=[]
    true_positive=[]
    # y_test = y_test.to_numpy()
    y_pred = y_pred.flatten()

    for i in range(len(y_pred)):
      if y_test[i] == y_pred[i]:
          correct += 1
          true_positive.append(i)
          if y_pred[i]==1:
              state_register_index.append(i)

    accuracy = correct/len(y_pred)
    state_register_accuracy = len(state_register_index)/list(y_test).count(1)

    return true_positive, state_register_index, state_register_accuracy, accuracy

if __name__ == '__main__':
    df_train = pd.read_csv('./data/runs_20200427-163557/combine_training_data.csv')
    df_train = df_train.drop([df_train.columns[13]], axis=1)
    scalar = StandardScaler()
    df_train = pd.DataFrame(scalar.fit_transform(df_train))
    df_train = df_train.drop([df_train.columns[1], df_train.columns[2], df_train.columns[4], df_train.columns[6], df_train.columns[7], df_train.columns[11]], axis=1)

    target_name = 'xx_state_ff'
    df_test = pd.read_csv('csvs/test/uart.csv')
    # df_test = pd.read_csv('csvs/b10_reset.csv')
    df_test = transpose_data(df_test)
    df_y_test = df_test.pop(target_name)
    df_test = pd.DataFrame(scalar.transform(df_test))
    df_test = df_test.drop([df_test.columns[1], df_test.columns[2], df_test.columns[4], df_test.columns[6], df_test.columns[7], df_test.columns[11]], axis=1)


    restore_accuracy = Restore('./data/runs_20200427-163557/model/trained_model_1_Data_1_val_accuracy_20200427-163557.zip')
    restore_loss = Restore('./data/runs_20200427-163557/model/trained_model_1_Data_1_val_loss_20200427-163557.zip')
    model_accuracy = restore_accuracy.model
    model_accuracy_details = restore_accuracy.details
    model_accuracy_params = restore_accuracy.params

    model_loss = restore_loss.model
    model_loss_details = restore_loss.details
    model_loss_params = restore_loss.params


    df_y_pred_acc = model_accuracy.predict_classes(df_test.to_numpy().astype(int))
    df_y_pred_loss = model_loss.predict_classes(df_test.to_numpy().astype(int))


    # df_y_pred = scalar.inverse_transform(df_y_pred)
    print("Actual Data: ", df_y_test.to_numpy())
    print("Predicted Data - Accuracy Model: ", df_y_pred_acc.flatten())
    print("Predicted Data - Loss Model: ", df_y_pred_loss.flatten())


    true_positive_acc, state_register_index_acc, state_register_accuracy_acc, model_accuracy_acc = true_positive_checker(df_y_test.to_numpy(), df_y_pred_acc)
    false_positive_acc = false_positive_checker(df_y_test, df_y_pred_acc)

    true_positive_loss, state_register_index_loss, state_register_accuracy_loss, model_loss_acc = true_positive_checker(df_y_test.to_numpy(), df_y_pred_loss)
    false_positive_loss = false_positive_checker(df_y_test, df_y_pred_loss)

    print("Accuracy Model - Index with True Positive: ", true_positive_acc)
    print("Accuracy Model - Index with False Positive: ", false_positive_acc)
    print("Accuracy Model - State Register Index: {}".format(state_register_index_acc))
    print("Accuracy Model - State Register Accuracy: {}".format(state_register_accuracy_acc))
    print("Accuracy Model - Accuracy = {}".format(model_accuracy_acc))
    # print(model_accuracy_details)
    # print(model_accuracy_params)

    print("Loss Model - Index with True Positive: ", true_positive_loss)
    print("Loss Model - Index with False Positive: ", false_positive_loss)
    print("Loss Model - State Register Index: {}".format(state_register_index_loss))
    print("Loss Model - State Register Accuracy: {}".format(state_register_accuracy_loss))
    print("Loss Model - Accuracy = {}".format(model_loss_acc))
    # print(model_loss_details)
    # print(model_loss_params)
