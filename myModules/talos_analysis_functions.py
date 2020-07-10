from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


def counter(df, df_feature_column_name, accuracy_target):
    accuracy_target_df = df.loc[df['val_accuracy']>accuracy_target]
    accuracy_target_df = accuracy_target_df.sort_values('val_accuracy', ascending=False, axis=0)
    accuracy_target_df.reset_index(drop=True, inplace=True)

    df_feature_column_vector = accuracy_target_df[df_feature_column_name]
    df_feature_column_vector = df_feature_column_vector.to_list()
    feature_set = set(df_feature_column_vector)
    feature_unique_val = list(feature_set)
    feature_unique_val.sort()
    print(df_feature_column_name)
    for i in feature_unique_val:
        counter = df_feature_column_vector.count(i)
        print("{}:".format(i), counter)
    print("\n")


def selector(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    range_hn = np.unique(df['hidden_neurons'].to_numpy())
    features = list(df.columns)
    features.remove(df.columns[1])
    features.remove(df.columns[2])
    features.remove(df.columns[3])
    features.remove(df.columns[4])

    z_axis_list = [df.columns[1], df.columns[2], df.columns[3], df.columns[4] ]
    z_axis = input("Select z-axis {}: ".format(z_axis_list).strip())
    # z_axis = "val_accuracy"
    while z_axis not in z_axis_list:
        z_axis = input("Select z-axis {}: ".format(z_axis_list).strip())

    print(features)
    x_axis, y_axis = input("Enter x-axis and y-axis, seperate by a single spacing: ").strip().split()
    # x_axis, y_axis = "hidden_layers", "hidden_neurons"
    while (x_axis or y_axis) not in features:
        x_axis, y_axis = input("Enter x-axis and y-axis, seperate by a single spacing: ").strip().split()
    features.remove(x_axis)
    features.remove(y_axis)

    features_drop = list(map(str, input("Select features to drop from list {}, seperate by a single spacing: ".format(features)).strip().split()))
    # features_drop = ["round_epochs", "activation", "last_activation"]
    df = df.drop(features_drop, axis=1)
    for i in features_drop:
        features.remove(i)
    print(features)


    fixed_features_input=[]
    print("Select fixed feature size")
    for f in features:
        feature_options = sorted((list(set(df[f]))))
        if f in ["activation", "last_activation", "loss.1", "optimizer"]:
            user_input = input("Select {} from list {}: ".format(f, feature_options))
        else:
            user_input = float(input("Select {} from list {}: ".format(f, feature_options)))

        while user_input not in feature_options:
            if f in ["activation", "last_activation", "loss.1", "optimizer"]:
                user_input = input("Select {} from list {}: ".format(f, feature_options))
            else:
                user_input = float(input("Select {} from list {}: ".format(f, feature_options)))

        fixed_features_input.append(user_input)

    for c, f in zip(features, fixed_features_input):
        print(c, f)
        df = df.loc[df[c] == f]
    df.reset_index(drop=True, inplace=True)
    print(df)

    for i in range(0,len(df),len(range_hn)):
        plotting_frame = df[i:i+len(range_hn)]
        x,y,z = plotting_frame[x_axis].to_numpy(), plotting_frame[y_axis].to_numpy(), plotting_frame[z_axis].to_numpy()
        ax.plot(x,y,z)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)
    plt.show()


def manual_average_selector(df):
    features = list(df.columns)
    features.remove(df.columns[1])
    features.remove(df.columns[2])
    features.remove(df.columns[3])
    features.remove(df.columns[4])

    print(features)
    param1 = input("Select parameter to fix: ")
    while param1 not in features:
        param1 = input("Select parameter to fix: ")
    param1_options = input("Select from list {}: ".format(sorted(list(set(df[param1])))))
    param1_options = param1_options if param1 in ["activation", "last_activation", "loss.1", "optimizer"] else float(param1_options)
    while param1_options not in sorted(list(set(df[param1]))):
        param1_options = input("Select from list {}: ".format(sorted(list(set(df[param1])))))
        param1_options = param1_options if param1 in ["activation", "last_activation", "loss.1", "optimizer"] else float(param1_options)
    features.remove(param1)
    df =  df.loc[df[param1]==param1_options]

    print(features)
    param2 = input("Select another parameter to fix: ")
    while param2 not in features:
        param2 = input("Select parameter to fix: ")
    param2_options = input("Select from list {}: ".format(sorted(list(set(df[param2])))))
    param2_options = param2_options if param2 in ["activation", "last_activation", "loss.1", "optimizer"] else float(param2_options)
    while param2_options not in sorted(list(set(df[param2]))):
        param2_options = input("Select from list {}: ".format(sorted(list(set(df[param2])))))
        param2_options = param2_options if param2 in ["activation", "last_activation", "loss.1", "optimizer"] else float(param2_options)
    features.remove(param2)
    df =  df.loc[df[param2]==param2_options]
    df.reset_index(drop=True, inplace=True)

    map_activation = dict([(activation_element, activation_index+1) for activation_index, activation_element in enumerate(sorted(set(df['activation'].tolist())))])
    map_last_activation = dict([(last_activation_element, last_activation_index+1) for last_activation_index, last_activation_element in enumerate(sorted(set(df['last_activation'].tolist())))])
    map_loss_fnc = dict([(loss_fn_element, loss_fn_index+1) for loss_fn_index, loss_fn_element in enumerate(sorted(set(df['loss.1'].tolist())))])
    map_optimizer = dict([(optimizer_element, optimizer_index+1) for optimizer_index, optimizer_element in enumerate(sorted(set(df['optimizer'].tolist())))])

    for key in map_activation:
        df=df.replace(key, map_activation.get(key))
    for key in map_last_activation:
        df=df.replace(key, map_last_activation.get(key))
    for key in map_loss_fnc:
        df=df.replace(key, map_loss_fnc.get(key))
    for key in map_optimizer:
        df=df.replace(key, map_optimizer.get(key))


    print(df)
    df =pd.DataFrame(df.mean(axis=0)).T
    print(df, type(df))


def auto_average_selector(df):
    fig = plt.figure()
    PLOTDIR = "./plot_data/"
    INTERACTIVE_PLOT_DIR = PLOTDIR + "interactive/"
    if not (os.path.exists(PLOTDIR) or os.path.exists(INTERACTIVE_PLOT_DIR)) :
        os.makedirs(INTERACTIVE_PLOT_DIR)

    df=df.drop(['activation', 'last_activation', 'loss', 'accuracy', 'loss.1', 'optimizer'], axis=1)
    features = list(df.columns)
    features.remove("val_loss")
    features.remove("val_accuracy")
    # print(features)
    # param1 = input("Select parameter to fix: ")
    # while param1 not in features:
    #     param1 = input("Select parameter to fix: ")
    # features.remove(param1)

    marker = ['.', '^', 's', 'p', 'P', '*', 'h', 'X', 'D']
    colour = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for x_axis in features:
        features_copy = features.copy()
        features_copy.remove(x_axis)
        for y_axis in features_copy:
            ax = fig.add_subplot(111, projection='3d')
            for  index, i in enumerate(np.unique(df[x_axis])):
                for j in np.unique(df[y_axis]):
                    df_copy = df.copy()
                    df_copy = df_copy.loc[(df_copy[x_axis]==i) & (df_copy[y_axis]==j)]
                    df_copy = pd.DataFrame(df_copy.mean(axis=0)).T

                    x,y,z= df_copy[x_axis].to_numpy(), df_copy[y_axis].to_numpy(), df_copy['val_accuracy'].to_numpy()
                    # ax.figure(index)
                    ax.scatter(x, y, z, marker=marker[index], c=colour[index])
            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_zlabel('val_accuracy')
            plt.savefig(PLOTDIR + "{}_{}.png".format(x_axis, y_axis))
            with open(INTERACTIVE_PLOT_DIR + "{}_{}.pickle".format(x_axis, y_axis),"wb") as pickle_out:
                pickle.dump(fig, pickle_out)
            plt.clf()

if __name__== "__main__":
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')

    pd.set_option('display.max_colwidth', None)
    # pd.set_option('display.max_rows', None)
    talos_data_df = pd.read_csv('~/Thesis/thesis_neural_network/data/runs_20200427-163557/model/100_Runs_Feature_Selection_Data_1/042720184056.csv')

    # auto_average_selector(talos_data_df)
    selector(talos_data_df)
    # counter(talos_data_df, "first_neuron", 0.82)
