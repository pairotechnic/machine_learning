# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np

# Local Application Imports
from machine_learning_specialization.supervised.classification.utils.plt_logistic_loss import  (
    plt_logistic_cost, plt_two_logistic_loss_curves, plt_simple_example, 
    soup_bowl, plt_logistic_squared_error
)

# Library Configurations
# %matplotlib widget # works only in Jupyter Notebook environments
plt.style.use(".\\machine_learning_specialization\\supervised\\utils\\deeplearning.mplstyle")


def main():

    # Squared Error Cost used in Linear Regression
    soup_bowl()
    plt.close('all')

    # Categorical Data used as training set for Logistic Regression
    x_train = np.array([0., 1, 2, 3, 4, 5],dtype=np.longdouble)
    y_train = np.array([0,  0, 0, 1, 1, 1],dtype=np.longdouble)
    plt_simple_example(x_train, y_train)
    plt.close('all')

    # Squared Error Cost for Logistic Regression
    # It isn't smooth, and has many local minima where Gradient Descent can get stuck
    plt_logistic_squared_error(x_train,y_train)
    plt.show()
    plt.close('all')

    # Loss curves for 2 categorical target values y=1 and y=0
    plt_two_logistic_loss_curves()
    plt.close('all')

    # Plot of cost vs parameters and log of cost vs parameters
    cst = plt_logistic_cost(x_train,y_train)
    plt.close('all')


if __name__ == "__main__":
    main()