# Standard Library Imports

# Third-Party Library Imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Local Application Imports
from machine_learning_specialization.supervised.regression.utils.lab_utils_common import dlc
from machine_learning_specialization.supervised.regression.utils.lab_utils_multi import  load_house_data

np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

def normalize_data(X_train):
    # Scale/normalize the training data
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_train)
    print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
    print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

    return X_norm


def create_and_fit_model_to_data(X_norm, y_train):
    # Create and fit the regression model
    sgdr = SGDRegressor(max_iter=1000)
    sgdr.fit(X_norm, y_train)
    print(sgdr)
    print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

    return sgdr


def get_parameters(sgdr):
    # View parameters associeted with normalized data
    w_norm = sgdr.coef_
    b_norm = sgdr.intercept_
    print(f"model parameters:                   w: {w_norm}, b:{b_norm}")
    print( "model parameters from previous lab: w: [110.56 -21.27 -32.71 -37.97], b: 363.16")

    return w_norm, b_norm


def make_predictions(sgdr, X_norm, w_norm, b_norm, y_train):
    # Make predictions
    # First, using sgdr.predict()
    y_pred_sgd = sgdr.predict(X_norm)
    # Second, using w,b. 
    y_pred = np.dot(X_norm, w_norm) + b_norm  
    print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

    print(f"Prediction on training set:\n{y_pred[:4]}" )
    print(f"Target values \n{y_train[:4]}")

    return y_pred


def plot_predictions_and_targets(X_train, y_train, X_features, y_pred):
    # plot predictions and targets vs original features    
    fig,ax=plt.subplots(1,4,figsize=(12,3),sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:,i],y_train, label = 'target')
        ax[i].set_xlabel(X_features[i])
        ax[i].scatter(X_train[:,i],y_pred,color=dlc["dlorange"], label = 'predict')
    ax[0].set_ylabel("Price"); ax[0].legend();
    fig.suptitle("target versus prediction using z-score normalized model")
    plt.show()


def main():
    # Load the dataset
    X_train, y_train = load_house_data()
    X_features = ['size(sqft)','bedrooms','floors','age']

    X_norm = normalize_data(X_train)
    sgdr = create_and_fit_model_to_data(X_norm, y_train)
    w_norm, b_norm = get_parameters(sgdr)
    y_pred = make_predictions(sgdr, X_norm, w_norm, b_norm, y_train)
    plot_predictions_and_targets(X_train, y_train, X_features, y_pred)


if __name__ == "__main__":
    main()