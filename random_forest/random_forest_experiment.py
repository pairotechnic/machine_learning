'''
    Description :
    Use Random Forest model to make better predictions than Decision Trees
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

# Step 1 : Analyze the csv of melbourne housing data
melbourne_file_path = "C:\Repositories\machine_learning\datasets\melb_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)
# print(melbourne_data.columns)
# print(melbourne_data.describe())

# Step 2 : Filter rows with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# print(filtered_melbourne_data.columns)
# print(filtered_melbourne_data.head())

# Step 3 : Identify the column/target you want to predict
# y = filtered_melbourne_data["Price"]
y = np.log1p(filtered_melbourne_data["Price"])

# print(f"Lowest price post clean: {min(y)}")
# print(f"Highest price post clean: {max(y)}")

def create_grid_cell(lat, lon, cell_size=0.01):
    lat_bin = round(lat // cell_size)
    lon_bin = round(lon // cell_size)
    return f"{lat_bin}_{lon_bin}"

filtered_melbourne_data['GridCell'] = filtered_melbourne_data.apply(
    lambda row: create_grid_cell(row['Lattitude'], row['Longtitude'], cell_size=0.01),
    axis=1
)

# Step 4 : Identify the feature columns based on which you want to make predictions on the target column
all_columns = ['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount']

melbourne_features = [
    'Suburb', 
    # 'Address', # High cardinality column, many unique values when encoded, hard to generalise
    'Rooms', 
    'Type', 
    'Method', 
    # 'SellerG', # High cardinality column, many unique values when encoded, hard to generalise
    'Date', # High cardinality column, many unique values when encoded, hard to generalise
    'Distance', 
    # 'Postcode', # Contains useful info, MAE increased when removed
    # 'Bedroom2', # Redundant, because of Rooms
    'Bathroom', 
    'Car',
    'Landsize', # Extremely important
    'BuildingArea', # Extremely important
    # 'YearBuilt', # Noise
    'CouncilArea', 
    'Lattitude',
    'Longtitude', 
    # 'Regionname', # Less useful, and overlaps with Suburbs, preventing clean tree splitting
    # 'Propertycount', # Overlapped with suburb, prevented clean splitting
    'GridCell'  # Derived from latitude and longitude
]

# Step 5 : Capture the details of only those features in X
X = filtered_melbourne_data[melbourne_features]

# Keep Date column just to extract features
X['YearSold'] = pd.to_datetime(X['Date'], dayfirst=True).dt.year
X['MonthSold'] = pd.to_datetime(X['Date'], dayfirst=True).dt.month

# Drop raw Date column
X = X.drop('Date', axis=1)

# Add a derived feature
# X['RoomsPerBuildingArea'] = X['Rooms'] / (X['BuildingArea'] + 1)  # add 1 to avoid division by zero
# X['RoomsPerLandsize'] = X['Rooms'] / (X['Landsize'] + 1)

# Clip outlier values
# X['Landsize'] = X['Landsize'].clip(upper=2000)
# X['BuildingArea'] = X['BuildingArea'].clip(upper=500)

# Step 6 : Encode categorical columns using pg.get_dummies()
X_encoded = pd.get_dummies(X)

# Step 7 : Encode the data into training and test sets
train_X, val_X, train_y, val_y = train_test_split(X_encoded, y,random_state = 0)

# Step 8 : Define a function to define mean absolute error
def get_mae(train_X, val_X, train_y, val_y):
    '''
        Define a RandomForestRegressor model
        Fit it on the training data, then make a prediction on the test data
        Then compare the prediction with the test target
    '''
    # model = RandomForestRegressor(random_state=0)
    model = RandomForestRegressor(
        n_estimators=1000, 
        max_depth=30, 
        # min_samples_split=2,
        # min_samples_leaf=1,
        # max_features='sqrt', # Or leave this out
        random_state=0
    )
    model.fit(train_X, train_y)
    prediction_val = model.predict(val_X)
    # mae = mean_absolute_error(val_y, prediction_val)
    mae = mean_absolute_error(np.expm1(val_y), np.expm1(prediction_val))
    return mae

# Step 9 : Calculate mean absolute error
mae = get_mae(train_X, val_X, train_y, val_y)
print(f"\nMean Absolue Error : {mae}")

#########################################

# Mean Absolue Error : 169941.32806388705
melbourne_features = [
    'Suburb', 
    # 'Address', # High cardinality column, many unique values when encoded, hard to generalise
    'Rooms', 
    'Type', 
    'Method', 
    # 'SellerG', # High cardinality column, many unique values when encoded, hard to generalise
    'Date', # High cardinality column, many unique values when encoded, hard to generalise
    'Distance', 
    # 'Postcode', # Contains useful info, MAE increased when removed
    # 'Bedroom2', # Redundant, because of Rooms
    'Bathroom', 
    'Car',
    'Landsize', 
    'BuildingArea', 
    # 'YearBuilt', # Noise
    'CouncilArea', 
    'Lattitude',
    'Longtitude', 
    # 'Regionname', # Less useful, and overlaps with Suburbs, preventing clean tree splitting
    # 'Propertycount' # Overlapped with suburb, prevented clean splitting
]

# Mean Absolue Error : 171853.44539106978
melbourne_features_01 = [
    'Suburb', 
    # 'Address', # High cardinality column, many unique values when encoded, hard to generalise
    'Rooms', 
    'Type', 
    'Method', 
    # 'SellerG', # High cardinality column, many unique values when encoded, hard to generalise
    # 'Date', # High cardinality column, many unique values when encoded, hard to generalise
    'Distance', 
    # 'Postcode', # Contains useful info, MAE increased when removed
    # 'Bedroom2', # Redundant, because of Rooms
    'Bathroom', 
    'Car',
    'Landsize', 
    'BuildingArea', 
    'YearBuilt', 
    'CouncilArea', 
    'Lattitude',
    'Longtitude', 
    # 'Regionname', # Less useful, and overlaps with Suburbs, preventing clean tree splitting
    # 'Propertycount'
]