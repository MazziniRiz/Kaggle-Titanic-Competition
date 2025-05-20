import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

#load data
X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

#Preprocessing pipeline to switch features into numerical data
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ('cat_encoder', OneHotEncoder(sparse_output=False)),
])

num_attributes = ["Age", "SibSp", "Parch", "Fare"]
cat_attributes = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attributes),
    ('cat', cat_pipeline, cat_attributes)
])

X = preprocess_pipeline.fit_transform(X_train)
y_train = X_train['Survived']

#Training the Classifier
model = SVC(gamma='auto')
model_scores = cross_val_score(model, X, y_train, cv=10)
model_scores.mean()

#Getting a mean of 82.5%, therefore closer to the top than with other models like K-nearest neighbour
