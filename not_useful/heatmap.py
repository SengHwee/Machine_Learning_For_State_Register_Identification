import seaborn as sns
import numpy as numpy
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


talos_data_df = pd.read_csv('~/Thesis/thesis_neural_network/data/runs_20200427-163557/model/100_Runs_Feature_Selection_Data_1/042720184056.csv')
talos_data_df = talos_data_df.drop(['activation', 'last_activation', 'loss', 'accuracy'], axis=1)
talos_data_df = talos_data_df.replace(to_replace ="adam", 
                 value ="1")
talos_data_df = talos_data_df.replace(to_replace ="RMSprop", 
                 value ="2")
talos_data_df = talos_data_df.replace(to_replace ="binary_crossentropy", 
                 value ="1")
talos_data_df = talos_data_df.replace(to_replace ="sparse_categorical_crossentropy", 
                 value ="2")

scalar = StandardScaler()
talos_data_df = scalar.fit_transform(talos_data_df) 

talos_data_df = pd.DataFrame(talos_data_df)
talos_data_df.rename(columns={talos_data_df.columns[0]:"round_epochs", 
							  talos_data_df.columns[1]:"val_loss",
							  talos_data_df.columns[2]:"val_accuracy",
							  talos_data_df.columns[3]:"batch_size",
							  talos_data_df.columns[4]:"dropout",
							  talos_data_df.columns[5]:"epochs",
							  talos_data_df.columns[6]:"first_neuron",
							  talos_data_df.columns[7]:"hidden_layers",
							  talos_data_df.columns[8]:"hidden_neurons",
							  talos_data_df.columns[9]:"loss_fun",
							  talos_data_df.columns[10]:"optimizer"}, inplace=True)

corr = pd.DataFrame(talos_data_df).corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot=True
)

plt.show()