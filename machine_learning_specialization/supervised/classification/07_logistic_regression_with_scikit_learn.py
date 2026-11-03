# Standard Library Imports

# Third-Party Library Imports
import numpy as np
from sklearn.linear_model import LogisticRegression

# Local Application Imports
from machine_learning_specialization.supervised.classification.utils.helpers import create_datasets

def main():
    X, y = create_datasets

    # Create the model, and fit it on the data
    lr_model = LogisticRegression()
    lr_model.fit(X, y)

    # Make predictions on the dataset
    y_pred = lr_model.predict(X)
    print("Prediction on training set:", y_pred)

    # Calculate accuracy of the model's predictions wrt the true labels
    print("Accuracy on training set:", lr_model.score(X, y))


if __name__ == "__main__":
    main()