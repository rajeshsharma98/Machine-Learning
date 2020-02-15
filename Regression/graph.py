import matplotlib.pyplot as plt

dataset = pd.read_csv("file-name")

plt.scatter(dataset['column_name_1'], dataset['column_name_2'], color = 'red')
#plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.show()
