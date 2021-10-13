from sklearn.preprocessing import minmax_scale
from termcolor import colored as cl
import csv
import pandas as pd
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

visas = pd.read_csv('visas_neu2.csv', delimiter=',')
for (columnName, columnData) in visas.iteritems():
    visas[columnName] = visas[columnName].astype('category')
    visas[columnName] = visas[columnName].cat.codes
# visas.to_csv("visas_Numerical.csv", index=False, header=False)
X_var = visas[['EMPLOYER_NAME', 'SOC_NAME', 'JOB_TITLE', 'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'WORKSITE_CITY',
               'WORKSITE_STATE_ABB', 'YEAR']].values
y_var = visas['CASE_STATUS'].values
X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size=0.3, random_state=0)
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)
yhat = neigh.predict(X_test)
print(cl("KNN Model Accuracy: ", attrs=['bold']), cl(round(accuracy_score(y_test, yhat) * 100, 2), attrs=['bold']),
      '\n')
print(classification_report(y_test, yhat))
empname = input("Introduzca Employer Name: ")
socname = input("Introduzca SOC Name: ")
jobtit = input("Introduzca Job Title: ")
Position = input("Indique si es trabajo de tiempo completo (Y) o (N): ")
Wage = input("Introduzca sueldo mensual: ")
City = input("Introduzca Ciudad donde se encuentra el trabajo: ")
State = input("Introduzca abbreviación del estado donde se encuentra el trabajo: ")
Yr = input("Introduzca año de aplicación: ")
with open('userinput.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['EMPLOYER_NAME', 'SOC_NAME', 'JOB_TITLE', 'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'WORKSITE_CITY', 'WORKSITE_STATE_ABB', 'YEAR'])
    filewriter.writerow([empname, socname, jobtit, Position, Wage, City, State, Yr])
datos = pd.read_csv('userinput.csv')
for (columnName, columnData) in datos.iteritems():
    datos[columnName] = datos[columnName].astype('category')
    datos[columnName] = datos[columnName].cat.codes
scaler = minmax_scale(datos)
print("Predicted Label: ", neigh.predict(scaler))
