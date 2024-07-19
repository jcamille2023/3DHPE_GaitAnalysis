import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
def main():
    data = pd.read_csv('melb_data.csv')
    y = data.Price
    X = data.copy().drop(['Price'],axis=1)
    X_t, X_v, y_t, y_v = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)
    categorical_cols = [cname for cname in X_t.columns if X_t[cname].nunique() < 10 and X_t[cname].dtype == "object"]
    numerical_cols = [cname for cname in X_t.columns if X_t[cname].dtype in ['int64','float64']]
    my_cols = categorical_cols + numerical_cols
    X_tr = X_t[my_cols].copy()
    X_va = X_v[my_cols].copy()

    my_pl = build_pipe(categorical_cols=categorical_cols,numerical_cols=numerical_cols)
    scores = -1 * cross_val_score(my_pl, X, y, cv=5, scoring='neg_mean_absolute_error')
    print("MAE scores: ",scores)
    print("Average MAE: ",scores.mean())
    return 0
def calcuate_mae(pipe, X_t, X_v, y_t, y_v):
    pipe.fit(X_t,y_t)
    preds = pipe.predict(X_v)
    print("Prediction 1: ", preds[0])
    print("Actual 1: ", y_v.iloc[0])
    return mean_absolute_error(y_v,preds)

def build_pipe(**kwargs):
    model = RandomForestRegressor(n_estimators=100, random_state=0) # builds the model
    n_si = SimpleImputer(strategy='constant') # imputes numerical data
    c_si = SimpleImputer(strategy='constant') # imputes categorical data

    if ('numerical_cols' in kwargs) and ("categorical_cols" in kwargs):
        c_t = Pipeline(steps=[
            ('imputer', c_si),
            ('onehot', OneHotEncoder(handle_unknown='ignore')) # one hot encodes categorical data
        ])
        preprocessor = ColumnTransformer(transformers=[ # wraps the imputers in a transformer
            ('num',n_si,kwargs['numerical_cols']),
            ('cat',c_t,kwargs['categorical_cols'])
        ])
    return Pipeline(steps=[
        ('preprocessor',preprocessor),
        ('model',model)
    ])
main()