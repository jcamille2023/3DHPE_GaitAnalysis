from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
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

    # features to take into account
    features = ["housing_median_age", "total_rooms", "total_bedrooms", "households", "median_income"]

    # scaling training data
    training_data["median_house_value"] = training_data["median_house_value"] / 100000 # units of 100k
    training_data["housing_median_age"] /= 100 # units of 100 years
    training_data["total_rooms"] /= 10000 # units of 10k
    training_data["total_bedrooms"] /= 1000
    training_data["households"] /= 1000
    training_data["median_income"] /= 10




    X = training_data[features]
    print(X.describe())
    y = training_data["median_house_value"]
    # creating training vs test data
    X_t, X_v, y_t, y_v = train_test_split(X, y, random_state=1)
    best_tree_size([5,50,48,500,670,5000], X_t, X_v, y_t, y_v)

def get_mae(max_leaf_nodes, X_t, X_v, y_t, y_v):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(X_t, y_t)
    predictions = model.predict(X_v)
    print("Prediction 1: ", predictions[0])
    print("Actual 1: ", y_v.iloc[0])
    mae = mean_absolute_error(y_v, predictions)
    return mae
def best_tree_size(arr, X_t, X_v, y_t, y_v):
    arr_mae = []
    for i in arr:
        mae = get_mae(i, X_t, X_v, y_t, y_v)
        arr_mae.append(mae)
        print("Mean Absolute Error: " + str(mae))
    best_tree_size = arr[arr_mae.index(min(arr_mae))]
    print("Best Tree Size: " + str(best_tree_size))
    return best_tree_size


main()