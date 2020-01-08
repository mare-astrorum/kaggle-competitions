# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 16:21:57 2019

@author: A
"""

#%% Load data

import numpy as np
import pandas as pd

# Set the display to show more rows & columns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Load data
train_data = pd.read_csv(#Path to file/train.csv")
#
#
test_data = pd.read_csv(#Path to file/test.csv")

#%% Explore the data

train_data.head
train_data.info() # Cabin, embarked and age appear to have missing values
train_data.describe()

import matplotlib.pyplot as plt
train_data.hist(bins=50, figsize=(20, 15))
plt.show()


#%% Drop values that don't seem to be important for predictions

X = train_data.drop(["Name", "Ticket"], axis=1)

#%% Change PassengerID to index

X = X.set_index("PassengerId")

#%% Explore NaN

X.info()

# Age appears to have missing values, so will drop them for now

#X = X.dropna(subset=["Age"])

# Perhaps filling with median works better

median = X["Age"].median()
X["Age"].fillna(median, inplace=True)
X.info()

X_age = X[["Age"]]

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(X_age)
X_age = imputer.transform(X_age)

X[["Age"]] = X_age

# Now do embarked, select mode as the fill value


X_embarked = X[["Embarked"]]

imputer = SimpleImputer(strategy="most_frequent")
X_embarked = imputer.fit_transform(X_embarked)

X[["Embarked"]] = X_embarked

#%% Change Cabin type to bool

X_cabin = X[["Cabin"]]
X_cabin = X_cabin.notnull().astype('int')


# Custom transformer for this:

from sklearn.base import BaseEstimator, TransformerMixin

class ColumnToOneNull(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        one_null_column = X[["Cabin"]].notnull().astype('int')
        return one_null_column

column_to_one_null = ColumnToOneNull()
X_cabin_attr = column_to_one_null.transform(X)

X[["Cabin"]] = X_cabin_attr

#%% Change the categorical variables to dummies

features = X[["Sex", "Embarked"]]
X_cat = pd.get_dummies(X[features])
X = X.join(X_cat)
X = X.drop(["Sex", "Embarked"], axis=1)

# Now with one hot encoder:

from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(features)
housing_cat_1hot

#%%
housing_cat_1hot.toarray()

#%%
cat_encoder.categories_





#%% Assign label to predict

y = X["Survived"]

#%% Drop 'Survived' because that is the label

X = X.drop("Survived", axis=1)


#%% Start model exploration with linear SVM

from sklearn.svm import LinearSVC

lin_svc_reg = LinearSVC()
lin_svc_reg.fit(X, y)

from sklearn.model_selection import cross_val_score

cross_val_score(lin_svc_reg, X, y, scoring="accuracy", cv=3)

# Not too epico array([0.64016736, 0.69747899, 0.79324895])


#%% Will try KNN

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier(n_neighbors = 28, weights='uniform')
knn_clf.fit(X, y)

cross_val_score(knn_clf, X, y, scoring="accuracy", cv=3)


#%% Do a Random Search to optimize parameters

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

param_grid = {'weights': ['uniform', 'distance'],
              'n_neighbors': sp_randint(10, 50),
              }

random_search = RandomizedSearchCV(knn_clf, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True, random_state=42, verbose=3)

random_search.fit(X, y)

random_search.best_params_
random_search.best_estimator_

#%% Will try RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
forest_clf.fit(X, y)


forest_scores = cross_val_score(forest_clf, X, y, scoring="accuracy", cv=10)
forest_scores.mean()


predictions = forest_clf.predict(test_data)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)


#%% OK, Random Forest works pretty well, will create a pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

age_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler())
        ])

age_tr = age_pipeline.fit_transform(X[["Age"]])



from sklearn.compose import ColumnTransformer 

age_attrib = ["Age"]
embarked_attr = ["Embarked"]
cabin_attr = ["Cabin"]
cat_attr = ["Sex", "Embarked"]

full_pipeline = ColumnTransformer([
        ("age", SimpleImputer(strategy="median"), age_attrib),
        ("embarked", SimpleImputer(strategy="most_frequent"), embarked_attr),
        ("cabin", ColumnToOneNull(), cabin_attr),
        ("cat", OneHotEncoder(), cat_attr)
        ])

X_prepared = full_pipeline.fit_transform(X)
X_df = pd.DataFrame(X_prepared)
X_df = X_df.drop(1, axis=1)
X_df = np.c_[X_df, X[["Pclass", "SibSp"]]]
X_df = pd.DataFrame(X_df)

#%% Start pipeline exploration with linear SVM

from sklearn.svm import LinearSVC

lin_svc_reg = LinearSVC()
lin_svc_reg.fit(X, y)

from sklearn.model_selection import cross_val_score

cross_val_score(forest_clf, X_df, y, scoring="accuracy", cv=3)

# Not too epico array([0.64016736, 0.69747899, 0.79324895])


#%% Final prep

y = train_data["Survived"]
X = train_data.drop(["Name", "Ticket", "PassengerId", "Survived"], axis=1)



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.base import BaseEstimator, TransformerMixin

class ColumnToOneNull(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[[self.column_name]].notnull().astype('int')
    
column_to_one_null = ColumnToOneNull("Cabin")
X = train_data
cabin = column_to_one_null.transform(X)


imputer = SimpleImputer(strategy="most_frequent")
X[["Embarked"]] = imputer.fit_transform(X[["Embarked"]])


 

from sklearn.compose import ColumnTransformer 

num_attrib = ["Age", "Fare"]
cat_attr = ["Pclass", "Sex", "SibSp", "Parch", "Cabin", "Embarked"]

kaggle_attr = ["Pclass", "Sex", "SibSp", "Parch"]

full_pipeline = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_attrib),
        ("cat", OneHotEncoder(), cat_attr)
        ])

kaggle_pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), kaggle_attr)
        ])

    
X_prepared = pd.DataFrame(full_pipeline.fit_transform(X))
Kaggle_prepared = pd.DataFrame(kaggle_pipeline.fit_transform(train_data))

#%% Initial Random Forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

forest_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

forest_scores = cross_val_score(forest_clf, X_prepared, y, scoring="accuracy", cv=10, verbose=True)
forest_scores.mean()


#%% Do a Random Search to optimize parameters

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint

param_grid = {'criterion': ['gini', 'entropy'],
              'n_estimators': sp_randint(10, 500),
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth': sp_randint(1, 30)
              }

random_search = RandomizedSearchCV(forest_clf, param_grid, cv=5, scoring="neg_mean_squared_error", return_train_score=True, random_state=42, verbose=3)

random_search.fit(X_prepared, y)

random_search.best_params_
random_search.best_estimator_

#%% Random Forest with the best params

forest_clf = RandomForestClassifier(n_estimators=199, max_depth=10, random_state=42, criterion = 'gini', max_features = 'log2')

forest_scores = cross_val_score(forest_clf, X_prepared, y, scoring="accuracy", cv=10, verbose=True)
forest_scores.mean()

forest_clf.fit(X_prepared, y)

predictions = forest_clf.predict(X_prepared)

#%% Will try KNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

knn_clf = KNeighborsClassifier(n_neighbors = 28, weights='uniform')
knn_clf.fit(X_prepared, y)

cross_val_score(knn_clf, X_prepared, y, scoring="accuracy", cv=3)

#%% TEST VALUES


X_test = test_data.drop(["Name", "Ticket", "PassengerId"], axis=1)

X_test[["Cabin"]] = column_to_one_null.transform(X)
X_test[["Embarked"]] = imputer.fit_transform(X_test[["Embarked"]])

X_test.info()
X_test.describe()


X_test_prepared = full_pipeline.fit_transform(X_test)

predictions = forest_clf.predict(X_test_prepared)
knn_predictions = knn_clf.predict(X_test_prepared)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': knn_predictions})
 
output.to_csv(#Path to file/my_submission.csv", index=False)



#%% Hands-on Machine Learning book code


# Get the files

import os

TITANIC_PATH = os.path.join("Data")


import pandas as pd

def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)

train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")


# Build a data frame selector

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]


# Pipeline for numerical attributes

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])
    

num_pipeline.fit_transform(train_data)


# Pipeline for categorical variables

from sklearn.preprocessing import OneHotEncoder



cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])


cat_pipeline.fit_transform(train_data)


# Join the numerical and categorical pipelines

from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
    


X_train = preprocess_pipeline.fit_transform(train_data)
X_train



y_train = train_data["Survived"]


# Use Random Forest 


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
forest_clf.fit(X_train, y_train)


# Predict test data

X_test_prepared = preprocess_pipeline.fit_transform(test_data)

final_predictions = forest_clf.predict(X_test_prepared)


output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': final_predictions})
output.to_csv(#Path to file/my_submission_bk.csv", index=False)

# OK, the kaggle.com submission is worse than with my attempt