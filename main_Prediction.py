import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

print("Script started.")
# Load dataset
data = pd.read_csv('train.csv', nrows=2000000)
print(data.head())

# Remove Impossible values & Outliers
def clean_data(df):
    df = df[df['fare_amount'].between(0, 200)]
    df = df[df['passenger_count'].between(1, 6)]
    
    df = df[
        df['pickup_latitude'].between(40, 42) &
        df['pickup_longitude'].between(-75, -72) &
        df['dropoff_latitude'].between(40, 42) &
        df['dropoff_longitude'].between(-75, -72)
    ]
    
    return df

# Calculate distance between pickup and dropoff points
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth's radius in km

    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi/2)**2 + \
        np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# Features
def add_features(df):
    # Convert to datetime
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    # Time features
    df['hour'] = df['pickup_datetime'].dt.hour
    df['weekday'] = df['pickup_datetime'].dt.weekday

     # Rush hour feature (binary or multiplier)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

    # Straight-line (Haversine) Distance
    df['distance_km'] = haversine_distance(
        df['pickup_latitude'], df['pickup_longitude'],
        df['dropoff_latitude'], df['dropoff_longitude']
    )

    # Simulated true driving distance
    df['true_distance_km'] = df['distance_km'] * 1.3

    # --- Manhattan zone encoding ---
    df['pickup_manhattan'] = (
        (df['pickup_latitude'].between(40.70, 40.88)) &
        (df['pickup_longitude'].between(-74.02, -73.92))
    ).astype(int)

    df['dropoff_manhattan'] = (
        (df['dropoff_latitude'].between(40.70, 40.88)) &
        (df['dropoff_longitude'].between(-74.02, -73.92))
    ).astype(int)

    # --- Estimated trip duration ---
    # average NYC taxi speed ~ 18 km/h
    avg_speed_kmh = 18
    df['est_trip_minutes'] = (df['true_distance_km'] / avg_speed_kmh) * 60

    return df

# Split Data into Train/Test
def split_data(df, features, target="fare_amount"):
    X = df[features]
    y = df[target]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
def train_linear_regression(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr

# Train Gradient Boosting
def train_gradient_boosting(X_train, y_train):
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gbr.fit(X_train, y_train)
    return gbr

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return mae, rmse

# Plot Feature Importances
def plot_feature_importance(model, feature_names):
    if hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        imp.sort_values().plot(kind='barh')
        plt.title("Feature Importance")
        plt.show()
    else:
        print("This model does not provide feature importances.")


# ------------------------------- #
# Pipeline Execution
# ------------------------------- #

print("Cleaning data...")
data = clean_data(data)

print("Adding features...")
data = add_features(data)

# List of features to use in the models
features = [
    "true_distance_km",
    "est_trip_minutes",
    "hour",
    "weekday",
    "passenger_count",
    "is_rush_hour",
    "pickup_manhattan",
    "dropoff_manhattan"
]

print("Splitting data...")
X_train, X_test, y_train, y_test = split_data(data, features)

print("Training Linear Regression...")
lr_model = train_linear_regression(X_train, y_train)

print("Training Gradient Boosting...")
gbr_model = train_gradient_boosting(X_train, y_train)

print("Evaluating models...")
lr_mae, lr_rmse = evaluate_model(lr_model, X_test, y_test)
gbr_mae, gbr_rmse = evaluate_model(gbr_model, X_test, y_test)

print("\n--- Results ---")
print(f"Linear Regression MAE: {lr_mae:.2f}, RMSE: {lr_rmse:.2f}")
print(f"Gradient Boosting MAE: {gbr_mae:.2f}, RMSE: {gbr_rmse:.2f}")

print("\nPlotting Feature Importance for Gradient Boosting...")
plot_feature_importance(gbr_model, features)
