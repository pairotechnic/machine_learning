# Standard Library Imports
import math

# Third-party Library Imports
import matplotlib.pyplot as plt
import numpy as np

# Local Application Imports
from machine_learning_specialization.supervised.regression.utils.lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

# Library Configurations
plt.style.use(".\\machine_learning_specialization\\supervised\\utils\\deeplearning.mplstyle")

# Function to calculate the cost
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2

    total_cost = 1 / (2*m) * cost
    return total_cost

def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
        x (ndarray (m,)): feature values - from m training examples
        y (ndarray (m,)): target values - from m training examples
        w, b (scalar)   : model parameters
    Returns :
        dj_dw (scalar): The gradient of the cost w.r.t the parameter w
        dj_db (scalar): The gradient of the cost w.r.t the parameter b 
    """

    # Number of training examples
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b

        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]

        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    """
    Performs gradient descent to fit w, b.
    Updates w, b by takingnum_iters gradient steps with learning rate alpha

    Args:
        x (ndarray (m,)): feature values - from m training examples
        y (ndarray (m,)): target values - from m training examples
        w_in, b_in (scalar): initial values of model parameters
        alpha (float): learning rate
        num_iters (int): number of iterations to run gradient descent
        cost_function: function to call to produce cost
        gradient_function: function to call to produce gradient

    Returns :
        w (scalar): Updated value of parameter after running gradient descent
        b (scalar): Updated value of parameter after running gradient descent
        J_history (List): History of cost values
        p_history (List): Hisotry of parameters [w, b]
    """

    # Arrays that store cost J and parameters w and b at each iteration for graphing later
    J_history = []
    p_history = []
    b = b_in
    w = w_in

    for i in range(num_iters):
        # Calculate the gradient and update the parameters using gradient_function
        dj_dw, dj_db = gradient_function(x, y, w, b)

        # Update the parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J and parameters w and b at each iteration
        if i<100000: # prevent resource exhaustion
            J_history.append(cost_function(x, y, w, b))
            p_history.append([w, b])

        # Print stats at every 10% of iterations, or every iteration if num_iters<10
        if i% math.ceil(num_iters/10) == 0:
            print(f"Iteration: {i:4}, Cost: {J_history[-1]:0.2e}, dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}, w: {w: 0.3e}, b: {b:0.5e}")

    return w, b, J_history, p_history # Return J_history and p_history for graphing

def main():
    # Load the data set
    x_train = np.array([1.0, 2.0]) # features
    y_train = np.array([300.0, 500.0]) # target value

    """
    Below, the left plot shows  ∂𝐽(𝑤,𝑏)∂𝑤 or the slope of the cost curve relative to 𝑤
    at three points. On the right side of the plot, the derivative is positive, while on the left it is negative. 
    Due to the 'bowl shape', the derivatives will always lead gradient descent toward the bottom 
    where the gradient is zero.

    The left plot has fixed  𝑏=100. Gradient descent will utilize both ∂𝐽(𝑤,𝑏)∂𝑤 and ∂𝐽(𝑤,𝑏)∂𝑏
    to update parameters. The 'quiver plot' on the right provides a means of viewing the gradient of both parameters. 
    The arrow sizes reflect the magnitude of the gradient at that point. The direction and slope of the arrow 
    reflects the ratio of ∂𝐽(𝑤,𝑏)∂𝑤 and ∂𝐽(𝑤,𝑏)∂𝑏 at that point. Note that the gradient points away from the minimum. 
    Review equation (3) above. The scaled gradient is subtracted from the current value of 𝑤 or 𝑏. 
    This moves the parameter in a direction that will reduce cost.
    """
    plt_gradients(x_train, y_train, compute_cost, compute_gradient)
    plt.show()

    # initialize parameters
    w_init = 0
    b_init = 0

    # some gradient descent settings
    iterations = 10000
    tmp_alpha = 1.0e-2

    # run gradient descent
    w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

    print(f"(w, b) found by gradient descent: ({w_final:8.4f}, {b_final:8.4f})")

    """
    Cost versus iterations of gradient descent
    A plot of cost versus iterations is a useful measure of progress in gradient descent. 
    Cost should always decrease in successful runs. The change in cost is so rapid initially, 
    it is useful to plot the initial decent on a different scale than the final descent. 
    In the plots below, note the scale of cost on the axes and the iteration step.
    """
    # plot cost versus iteration
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4)) 
    ax1.plot(J_hist[:100])
    ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
    ax1.set_title("Cost vs. iteration(start)")
    ax2.set_title("Cost vs. iteration(end)")
    ax1.set_ylabel('Cost')
    ax2.set_ylabel('Cost')
    ax1.set_xlabel('iteration step')
    ax2.set_xlabel('iteration step')
    plt.show()

    print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} thousand dollars")
    print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} thousand dollars")
    print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} thousand dollars")

    """
    You can show the progress of gradient descent during its execution, 
    by plotting the cost over iterations on a contour plot of the cost(w,b).

    The contour plot shows the  𝑐𝑜𝑠𝑡(𝑤,𝑏) over a range of 𝑤 and 𝑏. 
    Cost levels are represented by the rings. Overlayed, using red arrows, is the path of gradient descent. 
    
    Here are some things to note:
        The path makes steady (monotonic) progress toward its goal.
        initial steps are much larger than the steps near the goal.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))  
    plt_contour_wgrad(x_train, y_train, p_hist, ax) 
    plt.show()

    """
    Zooming in, we can see that final steps of gradient descent. 
    Note the distance between steps shrinks as the gradient approaches zero.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12,4)) 
    plt_contour_wgrad(x_train, y_train, p_hist, ax, w_range=[180, 220, 0.5], b_range=[80, 120, 0.5], contours=[1, 5, 10, 20], resolution=0.5) 
    plt.show()

    """
    The larger 𝛼 is, the faster gradient descent will converge to a solution. 
    But, if it is too large, gradient descent will diverge.
    """
    # initialize parameters
    w_init = 0
    b_init = 0

    # set alpha to a large value
    iterations = 10
    tmp_alpha = 8.0e-1

    """
    Here 𝑤 and 𝑏 are bouncing back and forth between positive and negative with the absolute value 
    increasing with each iteration. Further, each iteration ∂𝐽(𝑤,𝑏)∂𝑤 changes sign and 
    cost is increasing rather than decreasing. This is a clear sign that the learning rate is too large 
    and the solution is diverging. Let's visualize this with a plot.
    """
    # run gradient descent
    w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_init, b_init, tmp_alpha, iterations, compute_cost, compute_gradient)

    """
    Here, the left graph shows  𝑤's progression over the first few steps of gradient descent.  
    𝑤 oscillates from positive to negative and cost grows rapidly. Gradient Descent is operating on 
    both 𝑤 and 𝑏 simultaneously, so one needs the 3-D plot on the right for the complete picture.
    """
    plt_divergence(p_hist, J_hist, x_train, y_train)
    plt.show()

if __name__ == "__main__":
    main()