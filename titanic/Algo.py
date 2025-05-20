import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 

X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("ordinal_encoder", OrdinalEncoder()),
    ("imputer", SimpleImputer(strategy="most frequent")),
    ('cat_encoder', OneHotEncoder(sparse_output=False)),
])

num_attributes = ["Age", "SibSp", "Parch", "Fare"]
