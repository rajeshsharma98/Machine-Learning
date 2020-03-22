# K-Nearest Neighbors (K-NN)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, [1, 4]].values
y = dataset.iloc[:, -1].values

#Pairplots
import seaborn as sns
sns.pairplot(dataset.drop("Id", axis=1), hue="Species", size=3)  # Visualize the relationship between pairs of features

#Count Species Column-total number of index in which we have to clasify
dataset.Species.value_counts().plot(kind="pie", autopct='%.1f%%', figsize=(8,8))
dataset["Species"].value_counts() #without plot


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#le = LabelEncoder()  # Load the label encoder
#y = le.fit_transform(y)  # Encode the string target features into integers
encoder = OneHotEncoder(sparse=False)
encoder.fit_transform(y.reshape(-1,1))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred) * 100  # Compare accuracy of predicted classes with test data
print('k-Nearest Neighbours accuracy | ' + str(round(accuracy, 2)) + ' %.')


