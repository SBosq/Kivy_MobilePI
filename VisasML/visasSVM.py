import pandas as pd
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn import svm
from termcolor import colored as cl
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

visas = pd.read_csv('visas_neu2.csv', delimiter=',')
for (columnName, columnData) in visas.iteritems():
    visas[columnName] = visas[columnName].astype('category')
    visas[columnName] = visas[columnName].cat.codes
X_var = visas[['EMPLOYER_NAME', 'SOC_NAME', 'JOB_TITLE', 'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'WORKSITE_CITY',
               'WORKSITE_STATE_ABB', 'YEAR']].values
y_var = visas['CASE_STATUS'].values
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.3, random_state=0)
clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(cl("SVM Model Accuracy: ", attrs=['bold']), cl(round(accuracy_score(y_test, y_pred) * 100, 2), attrs=['bold']), '\n')
print(classification_report(y_test, y_pred, zero_division=0))
