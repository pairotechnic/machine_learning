'''
    Description : Use Decision Trees to make predictions using some dataset
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# Step 1 : Analyze the csv of transactions
upi_transactions_file_path = r"C:\Repositories\machine_learning\upi_transactions_2024.csv"
upi_transactions_data = pd.read_csv(upi_transactions_file_path)
# print(upi_transactions_data.columns)
# print(upi_transactions_data.describe())

# Step 2 : Filter out the rows with incomplete data
filtered_transactions_data = upi_transactions_data.dropna(axis=0)
# print(filtered_transactions_data.columns)
# print(filtered_transactions_data.head())

# Step 3 : Identify the column/target you want to predict
y = filtered_transactions_data['amount (INR)']

# Step 4 : Identify the relevant features based on which you want to make predictions of the target column
all_columns = ['transaction id', 'timestamp', 'transaction type', 'merchant_category',
       'amount (INR)', 'transaction_status', 'sender_age_group',
       'receiver_age_group', 'sender_state', 'sender_bank', 'receiver_bank',
       'device_type', 'network_type', 'fraud_flag', 'hour_of_day',
       'day_of_week', 'is_weekend']

transaction_features = [
    'transaction id', # Removing this improved performance a little
    # 'timestamp', # Removing this improved performance a little
    # 'transaction type', # Removing this improved performance a little
    'merchant_category', # Removing this reduced accuracy
    # 'amount (INR)', # Must be removed, because it's the target
    # 'transaction_status', # Removing this, same accuracy
    'sender_age_group', # Removing this reduced accuracy slightly
    # 'receiver_age_group', # Accuracy unchanged
    # 'sender_state', # Accuracy unchanged
    # 'sender_bank', # Accuracy Unchanged
    'receiver_bank', # Accuracy decreases very slightly
    'device_type', # Accuracy decreases very slightly
    # 'network_type', # Accuacy decreases slightly
    # 'fraud_flag', # Accuracy decreases slightly
    'hour_of_day',
    'day_of_week', 
    'is_weekend'
]

transaction_features_backup = [
    # 'transaction type', # Same
    'merchant_category', # Accuracy reduces without this
    'sender_age_group', # Accuracy reduces slightly
    # 'receiver_age_group', # Same
    # 'sender_state', # Same
    # 'sender_bank', # Same
    # 'receiver_bank', # Same
    # 'device_type', # Same
    # 'hour_of_day', # Same
    # 'day_of_week', # Accuracy increases without this
    # 'is_weekend' # Accuracy increases slightly without this
]

# Step 5 : Capture the data of only those features in X
X = filtered_transactions_data[transaction_features]

# Encode categorical columns
# X_encoded = pd.get_dummies(X)
X_encoded = X.copy()
label_encoders = {}

for col in X_encoded.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le

# Step 6 : Separate the encoded data into training and testing sets
train_X, val_X, train_y, val_y = train_test_split(X_encoded, y, random_state=0)

# Step 7 : Define a function to calculate mean absolute error, given max_leaf_nodes
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    prediction_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, prediction_val)
    return mae

# Step 7 : Calculate mean absolute error for various values of max_leaf_nodes
for max_leaf_nodes in [35, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 100]:
    mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print(f"Max Leaf Nodes : {max_leaf_nodes}, Mean Absolue Error : {mae}")

#############################################

# Max Leaf Nodes : 50, Mean Absolue Error : 913.6019998364519
transaction_features_01 = [
    'transaction id',
    'timestamp', 
    'transaction type', 
    'merchant_category',
    'transaction_status', 
    'sender_age_group',
    'receiver_age_group', 
    'sender_state', 
    'sender_bank', 
    'receiver_bank',
    'device_type', 
    'network_type', 
    'fraud_flag', 
    'hour_of_day',
    'day_of_week', 
    'is_weekend'
]

# Max Leaf Nodes : 51, Mean Absolue Error : 912.8683440719524
transaction_features = [
    'timestamp', 
    'transaction type', 
    'merchant_category',
    'transaction_status', 
    'sender_age_group',
    'receiver_age_group', 
    'sender_state', 
    'sender_bank', 
    'receiver_bank',
    'device_type', 
    'network_type', 
    'fraud_flag', 
    'hour_of_day',
    'day_of_week', 
    'is_weekend'
]

# Max Leaf Nodes : 52, Mean Absolue Error : 912.8086642107743
transaction_features_03 = [
    'transaction type', 
    'merchant_category',
    'transaction_status', 
    'sender_age_group',
    'receiver_age_group', 
    'sender_state', 
    'sender_bank', 
    'receiver_bank',
    'device_type', 
    'network_type', 
    'fraud_flag', 
    'hour_of_day',
    'day_of_week', 
    'is_weekend'
]

# Max Leaf Nodes : 28, Mean Absolue Error : 911.7766838787668
transaction_features = [
    'merchant_category',
    'transaction_status', 
    'sender_age_group',
    'receiver_age_group', 
    'sender_state', 
    'sender_bank', 
    'receiver_bank',
    'device_type', 
    'network_type', 
    'fraud_flag', 
    'hour_of_day',
    'day_of_week', 
    'is_weekend'
]

# Max Leaf Nodes : 46, Mean Absolue Error : 910.3244232175416
transaction_features = [
    'merchant_category', # Removing this reduced accuracy
    'sender_age_group', # Removing this reduced accuracy slightly
    'receiver_bank', # Accuracy decreases very slightly
    'device_type', # Accuracy decreases very slightly
    'network_type', # Accuacy decreases slightly
    'fraud_flag', # Accuracy decreases slightly
    'hour_of_day',
    'day_of_week', 
    'is_weekend'
]

#############################################

# Max Leaf Nodes : 23, Mean Absolue Error : 912.5837251163147
old_transaction_features_01 = [
    'transaction type', 'merchant_category','sender_age_group',
    'receiver_age_group', 'sender_state', 'sender_bank', 'receiver_bank',
    'device_type','hour_of_day','day_of_week', 'is_weekend'
]

# Max Leaf Nodes : 56, Mean Absolue Error : 909.3884938976726
old_transaction_features_02 = [
    'merchant_category', # Accuracy reduces without this
    'sender_age_group', # Accuracy reduces slightly
    'is_weekend'
]

# Max Leaf Nodes : 47, Mean Absolue Error : 909.3697444853739
old_transaction_features_03 = [
    'merchant_category', # Accuracy reduces without this
    'sender_age_group', # Accuracy reduces slightly
]

###################################################

# WRONG : Training data included target (price)
# Max Leaf Nodes : 9393, Mean Absolue Error : 0.566456
transaction_features = [
    'transaction id',
    'timestamp', 
    'transaction type', 
    'merchant_category',
    'amount (INR)', 
    'transaction_status', 
    'sender_age_group',
    'receiver_age_group', 
    'sender_state', 
    'sender_bank', 
    'receiver_bank',
    'device_type', 
    'network_type', 
    'fraud_flag', 
    'hour_of_day',
    'day_of_week', 
    'is_weekend'
]

#######################################