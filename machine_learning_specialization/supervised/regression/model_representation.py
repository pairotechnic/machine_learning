import numpy as np
import matplotlib.pyplot as plt
plt.style.use(".\\machine_learning_specialization\\supervised\\regression\\utils\\deeplearning.mplstyle")

def define_training_data_and_get_count():
    x_train = np.array([1.0, 2.0]) # input variable
    y_train = np.array([300.0, 500.0]) # target / output variable
    print(f"x_train = {x_train}")
    print(f"y_train = {y_train}")

    print(f"x_train.shape: {x_train.shape}")
    m = x_train.shape[0]
    # Alternatively, use the len function, since x_train is a 1-dimensional array
    # m = len(x_train)
    print(f"Number of training examples (m) : {m}")

    for i in range(m):
            x_i = x_train[i]
            y_i = y_train[i]
            print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

    return x_train, y_train, m

def compute_model_output(x, w, b):
        """
        Computes the prediction of a linear model
        Args:
            x (ndarray (m, )): Data, m examples
            w, b (scalar): model parameters
        Returns
            f_wb (ndarray (m, )): model prediction
        """
        m = x.shape[0]
        f_wb = np.zeros(m)
        for i in range(m):
            f_wb[i] = w * x[i] + b
        return f_wb

def plot_data(x_train, y_train, y_prediction=None):
    # Plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r', label='Actual Values')

    if y_prediction is not None:
        # Plot our model prediction
        plt.plot(x_train, y_prediction, c='b', label='Our Prediction')

    # Set the title
    plt.title("Housing Prices")

    # Set the y-axis label
    plt.ylabel("Price (in 100s of dollars)")

    # Set the x-axis label
    plt.xlabel("Size (1000 sq.ft.)")

    plt.legend() # Requires the label parameter be passed when calling plot or scatter functions

    plt.show()

def main():
    # Step 1 : Define training data points, and Identify 'm', i.e., the number of training examples
    x_train, y_train, m = define_training_data_and_get_count()

    plot_data(x_train, y_train)

    # Using the 2 training data points, I solved the linear equation y = wx + b
    w = 200
    b = 100
    print(f"w: {w}")
    print(f"b: {b}")

    y_prediction = compute_model_output(x_train, w, b, )

    plot_data(x_train, y_train, y_prediction)

    x_i = 1.2 # Since unit of measurement is 1000 sq.ft.
    cost_1200_sqft = w*x_i + b
    print(f"${cost_1200_sqft:.0f} thousand dollars")


if __name__ == "__main__":
    main()