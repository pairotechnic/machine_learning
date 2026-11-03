# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np

# Local Application Imports
from machine_learning_specialization.supervised.classification.utils.lab_utils_common import (
    plot_data, sigmoid, dlc
)

# Library Configurations

def create_datasets():
    X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
    y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)

    return X_train, y_train

def plot_datapoints(X_train, y_train):
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    plot_data(X_train, y_train, ax)

    # Set both axes to be from 0-4
    ax.axis([0, 4, 0, 3.5])
    ax.set_ylabel('$x_1$', fontsize=12)
    ax.set_xlabel('$x_0$', fontsize=12)
    plt.show()