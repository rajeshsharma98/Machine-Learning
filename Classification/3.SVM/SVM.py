#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[17]:


# Importing the dataset
dataset = pd.read_csv('Iris.csv')
X = dataset.iloc[:, [1, 4]].values
y = dataset.iloc[:, -1].values


# In[18]:


#Count Species Column-total number of index in which we have to clasify
dataset.Species.value_counts().plot(kind="pie", autopct='%.1f%%', figsize=(8,8))
dataset["Species"].value_counts() #without plot


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#le = LabelEncoder()  # Load the label encoder
#y = le.fit_transform(y)  # Encode the string target features into integers
encoder = OneHotEncoder(sparse=False)
encoder.fit_transform(y.reshape(-1,1))


# In[19]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[20]:



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[21]:


# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


# In[22]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[27]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred) * 100  # Compare accuracy of predicted classes with test data
print('SVM accuracy | ' + str(round(accuracy, 2)) + ' %.')



# In[29]:





# In[30]:




