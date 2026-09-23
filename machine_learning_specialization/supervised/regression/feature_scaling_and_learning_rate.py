# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np

# Local Application Imports
from machine_learning_specialization.supervised.regression.utils.lab_utils_common import dlc
from machine_learning_specialization.supervised.regression.utils.lab_utils_multi import  (
    load_house_data, run_gradient_descent, norm_plot, plt_equal_scale, plot_cost_i_w
)

# Library Configurations
plt.style.use(".\\machine_learning_specialization\\supervised\\utils\\deeplearning.mplstyle")
np.set_printoptions(precision=2)

def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column
    
    Args:
    X (ndarray (m,n))     : input data, m examples, n features
    
    Returns:
    X_norm (ndarray (m,n)): input normalized by column
    mu (ndarray (n,))     : mean of each feature
    sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)
    
def plot_each_feature_versus_price(X_train, y_train, X_features):
    fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("Price (1000's)")
    plt.show()

def normalization_process_illustration(X_train, X_features, X_mean, X_norm):
    fig,ax=plt.subplots(1, 3, figsize=(12, 3))
    ax[0].scatter(X_train[:,0], X_train[:,3])
    ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
    ax[0].set_title("unnormalized")
    ax[0].axis('equal')

    ax[1].scatter(X_mean[:,0], X_mean[:,3])
    ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
    ax[1].set_title(r"X - $\mu$")
    ax[1].axis('equal')

    ax[2].scatter(X_norm[:,0], X_norm[:,3])
    ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
    ax[2].set_title(r"Z-score normalized")
    ax[2].axis('equal')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("distribution of features before, during, after normalization")
    plt.show()

def feature_distribution_before_and_after_normalization(X_train, X_features, X_norm):
    fig,ax=plt.subplots(1, 4, figsize=(12, 3))
    for i in range(len(ax)):
        norm_plot(ax[i],X_train[:,i],)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("count");
    fig.suptitle("distribution of features before normalization")
    plt.show()
    fig,ax=plt.subplots(1,4,figsize=(12,3))
    for i in range(len(ax)):
        norm_plot(ax[i],X_norm[:,i],)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("count"); 
    fig.suptitle("distribution of features after normalization")

    plt.show()

def plot_target_and_zscore_prediction_against_original_features(X_norm, w_norm, b_norm, X_train, y_train, X_features):
    #predict target using normalized features
    m = X_norm.shape[0]
    yp = np.zeros(m)
    for i in range(m):
        yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features    
    fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train, label = 'target')
        ax[i].set_xlabel(X_features[i])
        ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
    ax[0].set_ylabel("Price"); ax[0].legend();
    fig.suptitle("target versus prediction using z-score normalized model")
    plt.show()

def normalize_and_predict_house_price(x_house, X_mu, X_sigma, w_norm, b_norm):
    # First, normalize out example using the mean and standard deviation derived when the training data
    x_house_norm = (x_house - X_mu) / X_sigma
    print(x_house_norm)
    x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
    print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")


def main():
    # load the dataset
    X_train, y_train = load_house_data()
    X_features = ['size(sqft)','bedrooms','floors','age']

    plot_each_feature_versus_price(X_train, y_train, X_features)

    # Run gradient descent with lower and lower alpha values till you find one where it doesn't diverge, 
    # and even while converging, it doesn't jump from side to side (oscillate around the minimum)
    sample_alpha_values = [9.9e-7, 9e-7, 1e-7]
    for sample_alpha in sample_alpha_values:
        _, _, hist = run_gradient_descent(X_train, y_train, 10, alpha = sample_alpha)
        plot_cost_i_w(X_train, y_train, hist)
    
    mu     = np.mean(X_train,axis=0)   
    sigma  = np.std(X_train,axis=0) 
    X_mean = (X_train - mu)
    X_norm = (X_train - mu)/sigma     

    normalization_process_illustration(X_train, X_features, X_mean, X_norm)

    # normalize the original features
    X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
    print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
    print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
    print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

    feature_distribution_before_and_after_normalization(X_train, X_features, X_norm)

    # Re-run gradient descent algorithm with normalized data. 
    # Note the vastly larger value of alpha. This will speed up gradient descent.
    w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )

    plot_target_and_zscore_prediction_against_original_features(X_norm, w_norm, b_norm, X_train, y_train, X_features)

    # Predict price of house that is not in training set (1200 sqft, 3 bedrooms, 1 floor, 40 years old) 
    x_house = np.array([1200, 3, 1, 40])
    normalize_and_predict_house_price(x_house, X_mu, X_sigma, w_norm, b_norm)

    # Plot cost contours side-by-side: unnormalized vs. z-score normalized features 
    # (shows why normalization speeds up gradient descent)
    plt_equal_scale(X_train, X_norm, y_train)

if __name__ == "__main__":
    main()