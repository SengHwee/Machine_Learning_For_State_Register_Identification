import pickle
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

with open("/home/senghwee/Thesis/thesis_neural_network/plot_data/interactive/batch_size_dropout.pickle", "rb") as pickle_in:
    fig = pickle.load(pickle_in)
plt.show()
