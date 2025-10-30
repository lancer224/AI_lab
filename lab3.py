import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn import tree
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
dt_regressor_model = DecisionTreeRegressor(max_depth=5, max_leaf_nodes=15)

from sklearn.tree import DecisionTreeClassifier
dt_classifier_model = DecisionTreeClassifier()

from sklearn.metrics import roc_curve



df = pd.read_csv(r"C:\Users\gobli\Documents\GitHub\AI_lab\lab1\PythonProject1\processed_titanic_train.csv")
X = df.drop(['Age'], axis = 1)
y = df['Age']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 38)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.3, random_state=38)
dt_regressor_model.fit(X_train, y_train)
y_pred_test = dt_regressor_model.predict(X_test)

MAE = mean_absolute_error(y_test, y_pred_test)
print("Regrission model")
print(" MAE: ", MAE)

names = X.columns
plt.figure(figsize=(20, 12))
tree.plot_tree(dt_regressor_model, feature_names=names, filled=True, rounded=True, fontsize=10, proportion=True)
plt.show()


Z = df.drop(['Transported'], axis = 1)
t = df['Transported']
Z_train, Z_test, t_train, t_test = train_test_split(Z, t, test_size = 0.35, random_state = 38)
Z_test, Z_val, t_test, t_val = train_test_split(Z_test, t_test, test_size=0.35, random_state=38)

dt_classifier_model.fit(Z_train, t_train)
t_proba = dt_classifier_model.predict_proba(Z_test)
fpr, tpr, threshholds = roc_curve(t_test, t_proba[:,1])

plt.plot(fpr, tpr, marker='o')
plt.ylim([0,1.1])
plt.xlim([0,1.1])
plt.ylabel('TRP')
plt.xlabel('FLP')
plt.title('ROC curve')
plt.show()

input()