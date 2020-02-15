#To predict pice of house according to the living area(square_feet)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing ther data
dataset = pd.read_csv('kc_house_data.csv')
X = np.array(dataset['sqft_living']).reshape(-1, 1)
y = np.array(dataset['price'])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Feature scaling- NOT REQUIRED

# Predicting the Test set results
pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Training set')
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Test Set')
plt.xlabel('Spce')
plt.ylabel('Price')
plt.show()

#To predict a value
#regressor.predict(np.array(numeric-value).reshape(1,-1))
