# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np

# Local Application Imports
from machine_learning_specialization.supervised.classification.utils.lab_utils_common import dlc, plot_data
from machine_learning_specialization.supervised.classification.utils.plt_one_addpt_onclick import plt_one_addpt_onclick

# Library Configurations
# %matplotlib widget # works only in Jupyter Notebook environments
plt.style.use(".\\machine_learning_specialization\\supervised\\utils\\deeplearning.mplstyle")

def create_datasets():
    # Define 1D and 2D datasets, with corresponding outputs
    x_train = np.array([0., 1, 2, 3, 4, 5])
    y_train = np.array([0,  0, 0, 1, 1, 1])
    X_train2 = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y_train2 = np.array([0, 0, 0, 1, 1, 1])

    return x_train, y_train, X_train2, y_train2


def plot_single_variable(ax, x_train, y_train):
    # Separate by class
    pos = y_train == 1
    neg = y_train == 0

    #plot 1, single variable
    ax.scatter(x_train[pos], y_train[pos], marker='x', s=80, c = 'red', label="y=1")
    ax.scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none', 
                edgecolors=dlc["dlblue"],lw=3)

    ax.set_ylim(-0.08,1.1)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('one variable plot')
    ax.legend()


def plot_two_variable(ax, X_train2, y_train2):
    #plot 2, two variables
    plot_data(X_train2, y_train2, ax)
    ax.axis([0, 4, 0, 4])
    ax.set_ylabel('$x_1$', fontsize=12)
    ax.set_xlabel('$x_0$', fontsize=12)
    ax.set_title('two variable plot')
    ax.legend()


def show_static_overview(x_train, y_train, X_train2, y_train2):
    """
    Builds the 1x2 figure with both static plots and displays it
    """

    fig, axes = plt.subplots(1,2,figsize=(8,3))

    plot_single_variable(axes[0], x_train, y_train)
    plot_two_variable(axes[1], X_train2, y_train2)

    plt.tight_layout()
    plt.show()


def run_interactive_regression(x_train, y_train):
    w_in = np.zeros((1))
    b_in = 0
    plt.close('all') 
    addpt = plt_one_addpt_onclick( x_train,y_train, w_in, b_in, logistic=False)
    plt.show()


def main():
    x_train, y_train, X_train2, y_train2 = create_datasets()
    show_static_overview(x_train, y_train, X_train2, y_train2)
    run_interactive_regression(x_train, y_train)


if __name__ == "__main__":
    main()