import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
random_classifier_model = RandomForestClassifier(max_depth=4, n_estimators=28, oob_score=True)

from sklearn.ensemble import AdaBoostClassifier
aba_classifier_model = AdaBoostClassifier(n_estimators=80)

from sklearn.ensemble import GradientBoostingClassifier
gradient_classifier_model = GradientBoostingClassifier(n_estimators=80)

df = pd.read_csv(r"C:\Users\gobli\Documents\GitHub\AI_lab\lab1\PythonProject1\processed_titanic_train.csv")
X = df.drop(['Transported'], axis = 1)
y = df['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 37)

random_classifier_model.fit(X, y)
print('OOBE: ', 1-random_classifier_model.oob_score_)

aba_classifier_model.fit(X_train, y_train)
print('ABAE: ', 1-aba_classifier_model.score(X_test, y_test))

gradient_classifier_model.fit(X_train, y_train)
print('GrBE: ', 1-gradient_classifier_model.score(X_test, y_test))

plt.figure(figsize=(12, 10))
y_proba = random_classifier_model.predict_proba(X_test)
fpr, tpr, threshholds = roc_curve(y_test, y_proba[:,1])
plt.plot(fpr, tpr, color='green')

y_proba = aba_classifier_model.predict_proba(X_test)
fpr, tpr, threshholds = roc_curve(y_test, y_proba[:,1])
plt.plot(fpr, tpr, color='blue')

y_proba = gradient_classifier_model.predict_proba(X_test)
fpr, tpr, threshholds = roc_curve(y_test, y_proba[:,1])
plt.plot(fpr, tpr, color='red')

plt.ylim([0,1.1])
plt.xlim([0,1.1])
plt.ylabel('TRP')
plt.xlabel('FLP')
plt.title('ROC curve')
plt.show()


