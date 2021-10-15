import pandas as pd
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning
from sklearn.preprocessing import MinMaxScaler
from termcolor import colored as cl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import warnings

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

visas1 = pd.read_csv('unified_Visas.csv', delimiter=',', dtype={'CASE_STATUS': 'str', 'JOB_TITLE': 'str',
                                                                'FULL_TIME_POSITION': 'str', 'EMPLOYER_NAME': 'str',
                                                                'EMPLOYER_STATE': 'str', 'WORKSITE_CITY_1': 'str',
                                                                'PREVAILING_WAGE_1': 'float'})
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
model = LogisticRegression(solver='saga', random_state=0, max_iter=5000).fit(X_var, y_var)
y_pred = model.predict(X_test)
print(cl("Logistic Regression Model Accuracy: ", attrs=['bold']),
      cl(round(accuracy_score(y_test, y_pred) * 100, 2), attrs=['bold']), '\n')
print(classification_report(y_test, y_pred))
