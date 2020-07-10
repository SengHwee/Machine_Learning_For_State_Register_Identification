from mpl_toolkits.mplot3d import axes3d
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

pd.set_option('display.max_colwidth', None)
df = pd.read_csv('042720184056.csv')

val_acc_70 = df.loc[df['val_accuracy']>0.82]
val_acc_70 = val_acc_70.sort_values('val_accuracy', ascending=False, axis=0)
val_acc_70.reset_index(drop=True, inplace=True)

print(val_acc_70)

def counter(df_feature):
    df_feature = df_feature.to_list()
    feature_set = set(df_feature)
    feature_unique_val = list(feature_set)
    feature_unique_val.sort()
    for i in feature_unique_val:
        counter = df_feature.count(i)
        print("{}:".format(i), counter)
    print("\n")

print('round_epochs')
counter(val_acc_70['round_epochs'])
print('batch_size')
counter(val_acc_70['batch_size'])
print('dropout')
counter(val_acc_70['dropout'])
print('epochs')
counter(val_acc_70['epochs'])
print('first_neuron')
counter(val_acc_70['first_neuron'])
print('hidden_layers')
counter(val_acc_70['hidden_layers'])
print('hidden_neurons')
counter(val_acc_70['hidden_neurons'])
print('loss_func')
counter(val_acc_70['loss.1'])
print('optimizer')
counter(val_acc_70['optimizer'])
