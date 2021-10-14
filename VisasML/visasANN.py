import warnings
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from termcolor import colored as cl
from collections import Counter
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
                'PREVAILING_WAGE_1']].values
y_var = visas1['CASE_STATUS'].values
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.3)
print(Counter(y_train))
print(Counter(y_test))
"""myscaler = StandardScaler()
myscaler.fit(X_train)
training_a = myscaler.transform(X_train)
testing_a = myscaler.transform(X_test)"""
m1 = MLPClassifier(hidden_layer_sizes=(4, 4, 4), activation='relu', random_state=0, verbose=True, learning_rate_init=0.3)
m1.fit(X_train, y_train)
y_pred = m1.predict(X_test)
ps0 = "{:.02f}".format(precision_score(y_test, y_pred, average='micro'))
ps1 = "{:.02f}".format(precision_score(y_test, y_pred, average='macro'))
rs0 = "{:.02f}".format(recall_score(y_test, y_pred, average='micro'))
rs1 = "{:.02f}".format(recall_score(y_test, y_pred, average='macro'))
fs0 = "{:.02f}".format(f1_score(y_test, y_pred, average='micro'))
fs1 = "{:.02f}".format(f1_score(y_test, y_pred, average='macro'))
data_dict = {'precision': [ps0, ps1], 'recall': [rs0, rs1], 'f1-score': [fs0, fs1]}
df = pd.DataFrame(data=data_dict, columns=['precision', 'recall', 'f1-score'])
print(cl("ANN Model Accuracy: ", attrs=['bold']), cl(round(accuracy_score(y_test, y_pred) * 100, 2), attrs=['bold']), '\n')
print(df)
