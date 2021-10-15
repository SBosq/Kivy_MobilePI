import warnings
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from termcolor import colored as cl
from collections import Counter

warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)

visas1 = pd.read_csv('visasD2.csv', delimiter=',', dtype={'CASE_STATUS': 'str', 'JOB_TITLE': 'str',
                                                          'FULL_TIME_POSITION': 'str', 'EMPLOYER_NAME': 'str',
                                                          'EMPLOYER_STATE': 'str', 'WORKSITE_CITY_1': 'str',
                                                          'PREVAILING_WAGE_1': 'float'})
visasNeu = visas1[(visas1 == 'CERTIFIED').any(axis=1)]
visasNeu1 = visas1[(visas1 == 'DENIED').any(axis=1)]
visasD1 = pd.read_csv('visasD1.csv')
visas1 = visas1.append(visasNeu1, ignore_index=True)
visas1 = visas1.append(visasD1, ignore_index=True)
for (columnName, columnData) in visas1.iteritems():
    visas1[columnName] = visas1[columnName].astype('category')
    visas1[columnName] = visas1[columnName].cat.codes
scaler = MinMaxScaler()
normVisas = pd.DataFrame(scaler.fit_transform(visas1), columns=visas1.columns, index=visas1.index)
# normVisas.to_csv("normVisas.csv", index=False)
X_var = normVisas[['JOB_TITLE', 'FULL_TIME_POSITION', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'WORKSITE_CITY_1',
                   'PREVAILING_WAGE_1']].values
y_var = normVisas['CASE_STATUS'].values
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.3)
m1 = MLPClassifier(hidden_layer_sizes=(6, 4), activation='identity', random_state=0, verbose=True,
                   learning_rate_init=0.3)
m1.fit(X_train, y_train)
y_pred = m1.predict(X_test)
print(cl("ANN Model Accuracy: ", attrs=['bold']), cl(round(accuracy_score(y_test, y_pred) * 100, 2), attrs=['bold']), '\n')
print(classification_report(y_test, y_pred))
