import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from datetime import datetime
import matplotlib.pyplot as plt
from joblib import dump, load  # Importing joblib for saving/loading models

# Load the dataset
file_path = r'expanded_energy_consumptions_dataset.xlsx'
df = pd.read_excel(file_path, sheet_name='Energy Consumptions')

# Feature Engineering
df['Illegal Consumption Electricity'] = df['Electricity Consumption'] - df['Expected Electricity Consumption']
df['% Over Expected Electricity'] = (df['Electricity Consumption'] / df['Expected Electricity Consumption']) * 100
df['Illegal Consumption Water'] = df['Water Consumption'] - df['Expected Water Consumption']
df['% Over Expected Water'] = (df['Water Consumption'] / df['Expected Water Consumption']) * 100
df['Illegal Consumption Gas'] = df['Gas Consumption'] - df['Expected Gas Consumption']
df['% Over Expected Gas'] = (df['Gas Consumption'] / df['Expected Gas Consumption']) * 100

# Add labels for whether consumption was illegal (1 if over the threshold, otherwise 0)
df['Illegal Electricity'] = (df['Illegal Consumption Electricity'] > 0).astype(int)
df['Illegal Water'] = (df['Illegal Consumption Water'] > 0).astype(int)
df['Illegal Gas'] = (df['Illegal Consumption Gas'] > 0).astype(int)

# Encode the 'Building' column
le_building = LabelEncoder()
df['Building_Encoded'] = le_building.fit_transform(df['Building'])

# Convert 'Date' to datetime and extract more features
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Quarter'] = df['Date'].dt.quarter
df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

# Select features for the model
features = ['Building_Encoded', 'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'IsWeekend',
            'Expected Electricity Consumption', 'Expected Water Consumption', 'Expected Gas Consumption',
            '% Over Expected Electricity', '% Over Expected Water', '% Over Expected Gas']

X = df[features]
y_water = df['Illegal Water']
y_electricity = df['Illegal Electricity']
y_gas = df['Illegal Gas']

# Function to create and train a model
def create_and_train_model(X, y, consumption_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"{consumption_type} Model Accuracy: {accuracy:.2f}%")
    
    # Plot true vs predicted values with date on the x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(df.loc[y_test.index, 'Date'], y_test, label='True Values', color='green', linestyle='-', marker='o')
    plt.plot(df.loc[y_test.index, 'Date'], y_pred, label='Predicted Values', color='red', linestyle='--', marker='x')
    plt.title(f'Comparison of True vs. Predicted {consumption_type} Consumption Over Time')
    plt.xlabel('Dates')
    plt.ylabel(f'{consumption_type} Consumption (0: Legal, 1: Illegal)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{consumption_type.lower()}_comparison.png')
    plt.show()
    
    return pipeline, accuracy

# Train models and print accuracies
water_model, water_accuracy = create_and_train_model(X, y_water, "Water")
electricity_model, electricity_accuracy = create_and_train_model(X, y_electricity, "Electricity")
gas_model, gas_accuracy = create_and_train_model(X, y_gas, "Gas")

# Save the trained models using joblib
dump(water_model, 'water_model.joblib')
dump(electricity_model, 'electricity_model.joblib')
dump(gas_model, 'gas_model.joblib')

# Function to make predictions
def make_prediction(model, input_data):
    return model.predict(input_data)[0]

# Get input from user
input_date = input("Enter the date for prediction (format: YYYY-MM-DD): ")
input_building = input("Enter the building name: ")

# Convert the date input
input_date = datetime.strptime(input_date, '%Y-%m-%d')
input_building_encoded = le_building.transform([input_building])[0]

# Prepare the input for prediction
input_data = pd.DataFrame({
    'Building_Encoded': [input_building_encoded],
    'Year': [input_date.year],
    'Month': [input_date.month],
    'Day': [input_date.day],
    'DayOfWeek': [input_date.weekday()],
    'Quarter': [(input_date.month - 1) // 3 + 1],
    'IsWeekend': [int(input_date.weekday() >= 5)],
    'Expected Electricity Consumption': [df['Expected Electricity Consumption'].mean()],
    'Expected Water Consumption': [df['Expected Water Consumption'].mean()],
    'Expected Gas Consumption': [df['Expected Gas Consumption'].mean()],
    '% Over Expected Electricity': [100],  # Default values, adjust if you have this information
    '% Over Expected Water': [100],
    '% Over Expected Gas': [100]
})

# Make predictions
pred_water = make_prediction(water_model, input_data)
pred_electricity = make_prediction(electricity_model, input_data)
pred_gas = make_prediction(gas_model, input_data)

# Output predictions
print(f"\nPredictions (1 means illegal, 0 means legal):")
print(f"Water: {pred_water}")
print(f"Electricity: {pred_electricity}")
print(f"Gas: {pred_gas}")

# Load the models (optional, if you want to demonstrate loading)
# water_model_loaded = load('water_model.joblib')
# electricity_model_loaded = load('electricity_model.joblib')
# gas_model_loaded = load('gas_model.joblib')
