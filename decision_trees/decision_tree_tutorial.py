import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

melbourne_file_path = "C:\Repositories\machine_learning\melb_data.csv"

melbourne_data = pd.read_csv(melbourne_file_path)

# print(melbourne_data.describe())

# print(melbourne_data.columns)

filtered_melbourne_data = melbourne_data.dropna(axis=0)

# Select the column/prediction target ( Our objective is to predict the value in this column )
y = filtered_melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# prices = filtered_melbourne_data["Price"]

# By convention, the data in the list of features is called X
X = filtered_melbourne_data[melbourne_features]
# print(X.describe())

# Inspect first few rows
# print(X.head())

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define model. Specify a number for random_state to ensure same results each run
# melbourne_model = DecisionTreeRegressor()

# melbourne_model.fit(X,y)

# print("Making predictions for the following 5 houses : ")
# print(X.head())

# print("The predictions are : ")
# print(melbourne_model.predict(X.head()))

# print(prices.head())

# predicted_home_prices = melbourne_model.predict(X)
# mae = mean_absolute_error(y, predicted_home_prices)

# print(f"Mean Absolute Error : {mae}")

# melbourne_model.fit(train_X, train_y)

# val_predictions = melbourne_model.predict(val_X)
# mae = mean_absolute_error(val_y, val_predictions)

# print(f"Mean Absolute Error : {mae}")

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(f"Max leaf nodes : {max_leaf_nodes}, Mean Absolute Error : {my_mae}")



print("EOF")