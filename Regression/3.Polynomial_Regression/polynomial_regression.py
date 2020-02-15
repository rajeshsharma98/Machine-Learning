import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#First create a fake dataset
from sklearn.datasets import make_regression
X, y = make_regression(n_samples = 300, n_features=1, noise=8, bias=2)
y2 = y**2

# Linear Regression- just to check
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y2)
plt.plot(X, model.predict(X))
plt.scatter(X, y2,color='red')
plt.title("Linear Model, Polynomial Degree = 1")
plt.show()

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures  
poly_features = PolynomialFeatures(degree = 3)  
X_poly = poly_features.fit_transform(X)
model.fit(X_poly, y2)

y_pred = model.predict(X_poly)
new_X, new_y = zip(*sorted(zip(X, y_pred))) # sort values for plotting
plt.plot(new_X, new_y)
plt.scatter(X,y2,color='red')
plt.title("Polynomial (Degree = 3)")
plt.show()
