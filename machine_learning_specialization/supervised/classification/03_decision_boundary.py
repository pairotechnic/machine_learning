# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np

# Local Application Imports
from machine_learning_specialization.supervised.classification.utils.lab_utils_common import plot_data, sigmoid, draw_vthresh

# Library Configurations
# %matplotlib widget # works only in Jupyter Notebook environments
plt.style.use(".\\machine_learning_specialization\\supervised\\utils\\deeplearning.mplstyle")

def create_datasets():
    X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1) 

    return X, y

def plot_datapoints(X, y):
    # Plot the data points with label y=1 as red crosses, 
    # while the data points with label y=0 as blue circles
    fig,ax = plt.subplots(1,1,figsize=(4,4))
    plot_data(X, y, ax)

    ax.axis([0, 4, 0, 3.5])
    ax.set_ylabel('$x_1$')
    ax.set_xlabel('$x_0$')
    plt.show()

def plot_z_against_sigmoid_z():
    # Plot sigmoid(z) over a range of values from -10 to 10
    z = np.arange(-10,11)

    fig,ax = plt.subplots(1,1,figsize=(5,3))
    # Plot z vs sigmoid(z)
    ax.plot(z, sigmoid(z), c="b")

    ax.set_title("Sigmoid function")
    ax.set_ylabel('sigmoid(z)')
    ax.set_xlabel('z')
    draw_vthresh(ax,0)

def plot_decision_boundary(X, y):
    # Choose values between 0 and 6
    x0 = np.arange(0,6)

    x1 = 3 - x0
    fig,ax = plt.subplots(1,1,figsize=(5,4))
    # Plot the decision boundary
    ax.plot(x0,x1, c="b")
    ax.axis([0, 4, 0, 3.5])

    # Fill the region below the line
    ax.fill_between(x0,x1, alpha=0.2)

    # Plot the original data
    plot_data(X,y,ax)
    ax.set_ylabel(r'$x_1$')
    ax.set_xlabel(r'$x_0$')
    plt.show()

def main():

    X, y = create_datasets()
    plot_datapoints(X, y)
    plot_z_against_sigmoid_z()
    plot_decision_boundary(X, y)

    

if __name__ == "__main__":
    main()