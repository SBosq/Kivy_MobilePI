from sklearn.preprocessing import MinMaxScaler
from termcolor import colored as cl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import tree
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

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
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.3, random_state=0)
treez = DecisionTreeClassifier(criterion="gini", max_depth=10, random_state=0).fit(X_train, y_train)
y_pred = treez.predict(X_test)
print(cl("Decision Tree Model Accuracy: ", attrs=['bold']),
      cl(round(accuracy_score(y_test, y_pred) * 100, 2), attrs=['bold']), '\n')
print(classification_report(y_test, y_pred))
"""num = list(map(str, input('Introduzca la informacion de su aplicacion: Es decir: Job Title, '
                          'tiempo completo (Y) o (N), Employer Name, Employer State, Employer City, sueldo, \n '
                          'Separados por una (,): ').split(',')))
print(num)
num = np.array(num).reshape(1, -1)
df = pd.DataFrame(num)
df.columns = ['JOB_TITLE', 'FULL_TIME_POSITION', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'WORKSITE_CITY_1',
              'PREVAILING_WAGE_1']
dataTrain = pd.read_csv('visas_Simplified.csv')
df1 = pd.DataFrame(dataTrain)
del df1['CASE_STATUS']
df1 = df1.append(df, ignore_index=True)
for (columnName, columnData) in df1.iteritems():
    df1[columnName] = df1[columnName].astype('category')
    df1[columnName] = df1[columnName].cat.codes
sample = df1.iloc[-1]
arr_sam = sample.to_numpy().reshape(1, -1)
print("Predicted Label: ", treez.predict(arr_sam))"""

"""fn = ['EMPLOYER_NAME', 'SOC_NAME', 'JOB_TITLE', 'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'WORKSITE_CITY',
      'WORKSITE_STATE_ABB', 'YEAR']

plt.figure(figsize=(25, 10))
tree.plot_tree(treez, feature_names=fn, class_names=['CERTIFIED', 'DENIED'], filled=True)
plt.show()"""
