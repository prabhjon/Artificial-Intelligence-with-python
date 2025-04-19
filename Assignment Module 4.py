import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

## Exercise 1: Regression to the mean
values_n = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]
for n in values_n:
    dice_throwing = np.random.randint(1, 7, size=(n, 2))
    addition = np.sum(dice_throwing, axis=1)
    h, h2 = np.histogram(addition, bins=range(2, 14))
    plt.bar(h2[:-1], h / n)
    plt.title(f'Histogram of Dice Sums for n={n}')
    plt.xlabel('Sum of Dice')
    plt.ylabel('Frequency')
    plt.show()

## Exercise 2: Regression Model
D1 = pd.read_csv('weight-height.csv')
plt.scatter(D1['Height'], D1['Weight'])
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Height vs Weight')
plt.show()
X = D1[['Height']]
y = D1['Weight']
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
plt.scatter(D1['Height'], D1['Weight'], label='Actual data')
plt.plot(D1['Height'], y_pred, color='red', label='Regression line')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Height Regression vs Weight Regression')
plt.legend()
plt.show()
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print(f'RMSE: {rmse}')
print(f'R2: {r2}')






