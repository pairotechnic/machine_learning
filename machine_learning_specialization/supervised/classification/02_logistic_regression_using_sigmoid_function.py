# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np

# Local Application Imports
from machine_learning_specialization.supervised.classification.utils.lab_utils_common import draw_vthresh
from machine_learning_specialization.supervised.classification.utils.plt_one_addpt_onclick import plt_one_addpt_onclick

# Library Configurations
# %matplotlib widget # works only in Jupyter Notebook environments
plt.style.use(".\\machine_learning_specialization\\supervised\\utils\\deeplearning.mplstyle")


def np_exp_function_demonstration():
    # Input is an array. 
    input_array = np.array([1,2,3])
    exp_array = np.exp(input_array)

    print("Input to exp:", input_array)
    print("Output of exp:", exp_array)

    # Input is a single number
    input_val = 1  
    exp_val = np.exp(input_val)

    print("Input to exp:", input_val)
    print("Output of exp:", exp_val)


def create_datasets():
    x_train = np.array([0., 1, 2, 3, 4, 5])
    y_train = np.array([0,  0, 0, 1, 1, 1])

    # Generate an array of evenly spaced values between -10 and 10
    z = np.arange(-10,11)

    return x_train, y_train, z


def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z

    Note: 
        NumPy performs arithmetic element-wise on arrays. 
        
        For example, if z = np.array([1, 2, 3]): 
            -z -> [-1, -2, -3] 
            np.exp(-z) -> [e^-1, e^-2, e^-3] 
            
        When a scalar is used with an array, NumPy broadcasts the scalar across every element: 
            1 + np.exp(-z) 
            -> [1, 1, 1] + [e^-1, e^-2, e^-3] 
            -> [1 + e^-1, 1 + e^-2, 1 + e^-3] 
                
        The division is also performed element-wise: 
            1 / (1 + np.exp(-z)) 
            -> [1/(1 + e^-1), 1/(1 + e^-2), 1/(1 + e^-3)]
        
    """

    g = 1/(1+np.exp(-z))

    return g


def plot_z_against_sigmoid_z(z, y):
    # Code for pretty printing the two arrays next to each other
    np.set_printoptions(precision=3) 
    print("Input (z), Output (sigmoid(z))")
    print(np.c_[z, y])

    # Plot z vs sigmoid(z)
    fig,ax = plt.subplots(1,1,figsize=(5,3))
    ax.plot(z, y, c="b")

    ax.set_title("Sigmoid function")
    ax.set_ylabel('sigmoid(z)')
    ax.set_xlabel('z')
    draw_vthresh(ax,0)


def run_interactive_regression(x_train, y_train):
    w_in = np.zeros((1))
    b_in = 0

    plt.close('all') 
    addpt = plt_one_addpt_onclick(x_train, y_train, w_in, b_in, logistic=True)
    plt.show()


def main():
    np_exp_function_demonstration()
    x_train, y_train, z = create_datasets()
    y = sigmoid(z)
    plot_z_against_sigmoid_z(z, y)
    run_interactive_regression(x_train, y_train)


if __name__ == "__main__":
    main()