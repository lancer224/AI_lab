import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import  LogisticRegression
logregmodel = LogisticRegression()
from sklearn.metrics import accuracy_score


df = pd.read_csv(r"C:\Users\gobli\Documents\GitHub\AI_lab\lab1\PythonProject1\processed_titanic_train.csv")
X = df.drop(['Age'], axis = 1)
y = df['Age']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 37)

linear_model.fit(X_train, y_train)
y_pred_test = linear_model.predict(X_test)

print("Linear")
MSE = mean_squared_error(y_test, y_pred_test)
print(" MSE: ", MSE)
RMSE = root_mean_squared_error(y_test, y_pred_test)
print("RMSE: ", RMSE)
MAE = mean_absolute_error(y_test, y_pred_test)
print(" MAE: ", MAE)

X = df.drop(['Transported'], axis = 1)
y = df['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 33)

logregmodel.fit(X_train, y_train)
y_pred_test = logregmodel.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_test)

print("Classification")
print(" ACC: ", accuracy)

input()


