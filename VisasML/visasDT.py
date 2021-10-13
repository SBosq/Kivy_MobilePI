from sklearn.preprocessing import minmax_scale
from termcolor import colored as cl
import csv
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

visas = pd.read_csv('visas_neu2.csv', delimiter=',')
for (columnName, columnData) in visas.iteritems():
    visas[columnName] = visas[columnName].astype('category')
    visas[columnName] = visas[columnName].cat.codes
X_var = visas[['EMPLOYER_NAME', 'SOC_NAME', 'JOB_TITLE', 'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'WORKSITE_CITY',
               'WORKSITE_STATE_ABB', 'YEAR']].values
y_var = visas['CASE_STATUS'].values
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.3, random_state=0)
treez = DecisionTreeClassifier(criterion="gini", max_depth=4, random_state=0).fit(X_train, y_train)

treez.predict(X_test)
y_pred = treez.predict(X_test)
print(cl("Decision Tree Model Accuracy: ", attrs=['bold']), cl(round(accuracy_score(y_test, y_pred) * 100, 2), attrs=['bold']), '\n')
print(classification_report(y_test, y_pred))

fn = ['EMPLOYER_NAME', 'SOC_NAME', 'JOB_TITLE', 'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'WORKSITE_CITY',
      'WORKSITE_STATE_ABB', 'YEAR']

plt.figure(figsize=(25, 10))
tree.plot_tree(treez, feature_names=fn, class_names=['CERTIFIED', 'DENIED'], filled=True)
plt.show()
