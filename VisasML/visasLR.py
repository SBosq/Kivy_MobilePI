import pandas as pd
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning
from termcolor import colored as cl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

visas1 = pd.read_csv('visas_Simplified.csv', delimiter=',', dtype={'CASE_STATUS': 'str', 'JOB_TITLE': 'str',
                                                                   'FULL_TIME_POSITION': 'str', 'EMPLOYER_NAME': 'str',
                                                                   'EMPLOYER_STATE': 'str', 'WORKSITE_CITY_1': 'str',
                                                                   'PREVAILING_WAGE_1': 'float'})
visasNeu = visas1[(visas1 == 'CERTIFIED').any(axis=1)]
visasNeu1 = visas1[(visas1 == 'DENIED').any(axis=1)]
visas1 = visasNeu.sample(frac=0.141)
visas1 = visas1.append(visasNeu1, ignore_index=True)
for (columnName, columnData) in visas1.iteritems():
    visas1[columnName] = visas1[columnName].astype('category')
    visas1[columnName] = visas1[columnName].cat.codes
X_var = visas1[['JOB_TITLE', 'FULL_TIME_POSITION', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'WORKSITE_CITY_1',
                'PREVAILING_WAGE_1']]
y_var = visas1['CASE_STATUS']
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.3, random_state=0)
model = LogisticRegression(solver='liblinear', random_state=0).fit(X_var, y_var)
y_pred = model.predict(X_test)
print("[0] precision_score: ", precision_score(y_test, y_pred, average='micro'))
print("[1] precision_score: ", precision_score(y_test, y_pred, average='macro'))
print("[0] recall_score: ", recall_score(y_test, y_pred, average='micro'))
print("[1] recall_score: ", recall_score(y_test, y_pred, average='macro'))
print("[0] F1_score: ", f1_score(y_test, y_pred, average='micro'))
print("[1] F1_score: ", f1_score(y_test, y_pred, average='macro'))
print(cl("Logistic Regression Model Accuracy: ", attrs=['bold']), cl(round(accuracy_score(y_test, y_pred) * 100, 2), attrs=['bold']), '\n')

