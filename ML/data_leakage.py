from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
def main():
    # Create a pipeline
    pipe = make_pipeline(SimpleImputer(strategy='constant'), XGBClassifier(n_estimators=1000,learning_rate=0.12,random_state=0))
    # Cross-validate the pipeline
    scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')
    print(scores)
    print(scores.mean())