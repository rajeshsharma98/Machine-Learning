import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

nyc_pumpkins = pd.read_csv("new-york_9-24-2016_9-30-2017.csv")
cat_map = {
    'sml': 0,
    'med': 1,
    'med-lge': 2,
    'lge': 3,
    'xlge': 4,
    'exjbo': 5
}
nyc_pumpkins = nyc_pumpkins.assign(
    size=nyc_pumpkins['Item Size'].map(cat_map),
    price=nyc_pumpkins['High Price'] + nyc_pumpkins['Low Price'] / 2,
    size_class=(nyc_pumpkins['Item Size'].map(cat_map) >= 2).astype(int)
)
nyc_pumpkins = nyc_pumpkins.drop([c for c in nyc_pumpkins.columns if c not in ['size', 'price', 'size_class']],
                                 axis='columns')
nyc_pumpkins = nyc_pumpkins.dropna()
sizes = nyc_pumpkins.values[:, :1]
prices = nyc_pumpkins.values[:, 1:2]

X = prices
y = sizes


from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(np.array(250).reshape(1,-1))
print(y_pred)

# Visualising the Decision Tree Regression results (higher resolution)
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.show()
