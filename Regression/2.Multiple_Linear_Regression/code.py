#To predict pice of house in accordanceto more than one opther fields
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing ther data
dataset = pd.read_csv('kc_house_data.csv')
X = dataset.iloc[:, 3:].values
y = dataset.iloc[:, 2].values

# Encoding categorical data
#There is no categorical data so encoding is not required

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
# Not Required

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
