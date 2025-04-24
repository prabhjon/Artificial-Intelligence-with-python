# Problem:diabetes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
data = load_diabetes(as_frame=True)
df = data['frame']
print(data.DESCR)
print(df.head())
plt.hist(df["target"], 25)
plt.xlabel("target")
plt.title("Target Distribution")
plt.show()
sns.heatmap(df.corr().round(2), annot=True)
plt.title("Correlation Matrix")
plt.show()
plt.subplot(1, 2, 1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel('bmi')
plt.ylabel('target')
plt.subplot(1, 2, 2)
plt.scatter(df['s5'], df['target'])
plt.xlabel('s5')
plt.ylabel('target')
plt.tight_layout()
plt.show()
X_base = df[['bmi', 's5']]
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=0.2, random_state=5)
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
rmse_base = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_base = r2_score(y_train, y_train_pred)
rmse_base_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_base_test = r2_score(y_test, y_test_pred)
print("Base RMSE Train:", rmse_base, "R2:", r2_base)
print("Base RMSE Test:", rmse_base_test, "R2:", r2_base_test)
"""
a) We add 'bp' because blood pressure is medically relevant and moderately correlated with target.
"""

X_bp = df[['bmi', 's5', 'bp']]

X_train, X_test, y_train, y_test = train_test_split(X_bp, y, test_size=0.2, random_state=5)

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

rmse_bp = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_bp = r2_score(y_test, y_test_pred)

print("With bp RMSE Test:", rmse_bp, "R2:", r2_bp)

"""
b) Adding 'bp' slightly improves model performance with lower RMSE and higher R2.
"""

X_all = df.drop(columns=["target"])
X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.2, random_state=5)

model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

rmse_all = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_all = r2_score(y_test, y_test_pred)

print("All Features RMSE Test:", rmse_all, "R2:", r2_all)

"""
d) Using all variables improves prediction the most with highest R2 and lowest RMSE.
"""
# Problem: Profit prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("50_Startups.csv")
print(df.head())
"""
Dataset contains R&D Spend, Administration, Marketing Spend, State, and Profit.
"""
df_encoded = pd.get_dummies(df, drop_first=True)
sns.heatmap(df_encoded.corr().round(2), annot=True)
plt.title("Correlation Matrix")
plt.show()
"""
R&D Spend has strongest positive correlation with profit, followed by Marketing Spend.
"""
plt.subplot(1, 2, 1)
plt.scatter(df['R&D Spend'], df['Profit'])
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.subplot(1, 2, 2)
plt.scatter(df['Marketing Spend'], df['Profit'])
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.tight_layout()
plt.show()
X = df[['R&D Spend', 'Marketing Spend']]
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
model = LinearRegression()
model.fit(X_train, y_train)
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)
print("Profit Train RMSE:", rmse_train, "R2:", r2_train)
print("Profit Test RMSE:", rmse_test, "R2:", r2_test)
"""
Profit prediction performs best using R&D Spend and Marketing Spend due to high correlation.
"""
# problem:car mpg
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
auto = pd.read_csv("Auto.csv", na_values="?").dropna()
print(auto.head())
X = auto.drop(columns=["mpg", "name", "origin"])
y = auto["mpg"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
alphas = np.logspace(-3, 2, 50)
r2_ridge = []
r2_lasso = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    r2_ridge.append(ridge.score(X_test, y_test))

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    r2_lasso.append(lasso.score(X_test, y_test))

plt.plot(alphas, r2_ridge, label='Ridge')
plt.plot(alphas, r2_lasso, label='Lasso')
plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('R2 Score')
plt.title('R2 Score vs Alpha')
plt.legend()
plt.show()
best_alpha_ridge = alphas[np.argmax(r2_ridge)]
best_alpha_lasso = alphas[np.argmax(r2_lasso)]  #
print("Best Ridge alpha:", best_alpha_ridge, "R2:", max(r2_ridge))
print("Best Lasso alpha:", best_alpha_lasso, "R2:", max(r2_lasso))
"""
Best alpha for Ridge and Lasso found by plotting and selecting value with highest R2 score.
"""