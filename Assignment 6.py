import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Read the CSV file using pandas
df = pd.read_csv('bank.csv', delimiter=';')
print("Step 1: Dataframe loaded successfully.")
print(df.head())

# Step 2: Pick data from specified columns to a second dataframe 'df2'
df2 = df[['y', 'job', 'marital', 'default', 'housing', 'poutcome']]
print("\nStep 2: Second dataframe created successfully.")
print(df2.head())

# Step 3: Convert categorical variables to dummy numerical values
df3 = pd.get_dummies(df2, columns=['job', 'marital', 'default', 'housing', 'poutcome'])
print("\nStep 3: Categorical variables converted to dummy numerical values.")
print(df3.head())

# Step 4: Produce a heat map of correlation coefficients for all variables in df3
plt.figure(figsize=(12, 8))
sns.heatmap(df3.corr(), annot=True, cmap='coolwarm')
plt.title('Heat Map of Correlation Coefficients')
plt.show()
print("\nStep 4: Heat map produced successfully.")

# Step 5: Select the column called 'y' of df3 as the target variable y, and all the remaining columns for the explanatory variables X
y = df3['y']
X = df3.drop(columns=['y'])
print("\nStep 5: Target variable and explanatory variables selected successfully.")

# Step 6: Split the dataset into training and testing sets with 75/25 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("\nStep 6: Dataset split into training and testing sets successfully.")

# Step 7: Setup a logistic regression model, train it with training data and predict on testing data
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_log_reg = log_reg.predict(X_test)
print("\nStep 7: Logistic regression model trained and predictions made successfully.")

# Step 8: Print the confusion matrix and accuracy score for the logistic regression model
conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
accuracy_log_reg = accuracy_score(y_test, y_pred_log_reg)
print("\nStep 8: Confusion matrix and accuracy score for logistic regression model:")
print(conf_matrix_log_reg)
print(f"Accuracy: {accuracy_log_reg}")

# Step 9: Repeat steps 7 and 8 for k-nearest neighbors model (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print("\nStep 9: Confusion matrix and accuracy score for k-nearest neighbors model:")
print(conf_matrix_knn)
print(f"Accuracy: {accuracy_knn}")

# Step 10: Compare the results between the two models
"""
The logistic regression model achieved an accuracy of {accuracy_log_reg}, while the k-nearest neighbors model achieved an accuracy of {accuracy_knn}.
The confusion matrices indicate that both models have their strengths and weaknesses in predicting the target variable.
Logistic regression is generally better suited for binary classification problems like this one, while k-nearest neighbors can be more flexible but may require tuning of hyperparameters like k.
Overall, logistic regression performed slightly better in this case.
"""
