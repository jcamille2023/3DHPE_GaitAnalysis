from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.impute import SimpleImputer

def main():
    # pandas configuration
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 30)
    training_data = pd.read_csv('melb_data.csv')
    # correlation matrix and data description
    print("Matrix:")
    #print(training_data.corr())
    print("Description:")
    print(training_data.describe())
    # features to take into account

    X = training_data
    X['Rooms'] /= 10
    X['Bathroom'] /= 10
    X['Landsize'] /= 1000
    X['BuildingArea'] /= 1000
    X['YearBuilt'] /= 1000


    print(X.describe())
    y = training_data["Price"] / (1*10**6)
    # creating training vs test data
    X_t, X_v, y_t, y_v = train_test_split(X, y, random_state=1)
    # categorical columns
    s = (X_t.dtypes == 'object')
    object_cols = list(s[s].index)
    print(object_cols)
    #copying data to keep original data
    X_t_c = X_t.copy()
    X_v_c = X_v.copy()
    #one hot encoding
    OH_encoder = OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    X_t_oh = pd.DataFrame(OH_encoder.fit_transform(X_t[object_cols]))
    X_v_oh = pd.DataFrame(OH_encoder.transform(X_v[object_cols]))
    print(X_t_oh.head())
    # One-hot encoding removed index; put it back
    X_t_oh.index = X_t_c.index
    X_v_oh.index = X_v_c.index

    # Imputation
    X_t_oh = impute_data(X_t)
    X_v_oh = impute_data(X_v)
    
    print("MAE: ", calcuate_mae(X_t_oh, X_v_oh, y_t, y_v))

    return 0

def impute_data(X):
    my_imputer = SimpleImputer()
    imputed_X = pd.DataFrame(my_imputer.fit_transform(X))
    imputed_X.columns = X.columns
    return imputed_X
def calcuate_mae(X_t, X_v, y_t, y_v):
    model = RandomForestRegressor(random_state=1)
    model.fit(X_t, y_t)
    predictions = model.predict(X_v)
    print("Prediction 1: ", predictions[300])
    print("Actual 1: ", y_v.iloc[300])
    mae = mean_absolute_error(y_v, predictions)
    return mae
main()