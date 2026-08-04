# Standard Library Imports

# Third-party Library Imports
import numpy as np
# matplotlib.use("TkAgg") # or "QtAgg"
import matplotlib
import matplotlib.pyplot as plt
print(f"matplotlib backend : {matplotlib.get_backend()}")
# works only in Jupyter Notebook environments
# %matplotlib widget 

# Local Application Imports
plt.style.use(".\\machine_learning_specialization\\supervised\\regression\\utils\\deeplearning.mplstyle")
from machine_learning_specialization.supervised.regression.utils.lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray (m, )): Input values, m examples
        y (ndarray (m, )): Target values, m examples
        w, b (scalar): model parameters

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w*x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost

    total_cost = (1/(2*m)) * cost_sum

    return total_cost

def main():

    x_train = np.array([1.0, 2.0]) # unit of size is 1000 sq.ft.
    y_train = np.array([300.0, 500.0]) # unit of price is 1000 dollars
    
    # Comes from a local import in the course's Cost Function Jupyter Notebook 
    # b is fixed, only w variable, creates 2d cost function
    plt_intuition(x_train,y_train)

    # Larger data set that doesn't exactly fit a line
    x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
    y_train = np.array([250, 300, 480, 430, 630, 730,])

    # w and b both variables
    # creates 3d cost function
    # approximately  𝑤=209 and  𝑏=2.4 provide lowest cost
    plt.close('all')
    fig, ax, dyn_items = plt_stationary(x_train, y_train)
    updater = plt_update_onclick(fig, ax, x_train, y_train, dyn_items)

    # Another 3d surface plot (Convex cost durface)
    soup_bowl()

if __name__ == "__main__":
    main()