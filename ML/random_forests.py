from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # pandas configuration
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 30)
    training_data = pd.read_csv('train.csv')

    # correlation matrix and data description
    print("Matrix:")
    print(training_data.corr())
    print("Description:")
    print(training_data.describe())

    # scaling training data
    training_data["median_house_value"] = training_data["median_house_value"] / 100000  # units of 100k
    training_data["housing_median_age"] /= 100  # units of 100 years
    training_data["total_rooms"] /= 10000  # units of 10k
    training_data["total_bedrooms"] /= 1000
    training_data["households"] /= 1000
    training_data["median_income"] /= 10
    print("Matrix:")
    print(training_data.corr())
    # features to take into account
    features = ["housing_median_age", "total_rooms", "total_bedrooms", "households", "median_income"]
    X = training_data[features]
    print(X.describe())
    y = training_data["median_house_value"]
    # creating training vs test data
    X_t, X_v, y_t, y_v = train_test_split(X, y, random_state=1)
    print("MAE: ", get_mae(X_t, X_v, y_t, y_v))

def get_mae(X_t, X_v, y_t, y_v):
    model = RandomForestRegressor(random_state=1)
    model.fit(X_t, y_t)
    predictions = model.predict(X_v)
    print("Prediction 1: ", predictions[0])
    print("Actual 1: ", y_v.iloc[0])
    mae = mean_absolute_error(y_v, predictions)
    return mae



main()