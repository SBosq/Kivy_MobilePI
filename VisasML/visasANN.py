import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from termcolor import colored as cl

visas = pd.read_csv('visas_neu2.csv', delimiter=',')
for (columnName, columnData) in visas.iteritems():
    visas[columnName] = visas[columnName].astype('category')
    visas[columnName] = visas[columnName].cat.codes

le = preprocessing.LabelEncoder()

X_var = visas[['EMPLOYER_NAME', 'SOC_NAME', 'JOB_TITLE', 'FULL_TIME_POSITION', 'PREVAILING_WAGE', 'WORKSITE_CITY',
               'WORKSITE_STATE_ABB', 'YEAR']].values
y_var = visas['CASE_STATUS'].values
training_a, testing_a, training_b, testing_b = train_test_split(X_var, y_var, test_size=0.3)
myscaler = StandardScaler()
myscaler.fit(training_a)
training_a = myscaler.transform(training_a)
testing_a = myscaler.transform(testing_a)
m1 = MLPClassifier(hidden_layer_sizes=(6, 5), random_state=0, verbose=True, learning_rate_init=0.3)
m1.fit(training_a, training_b)
predicted_values = m1.predict(testing_a)
print(cl("ANN Model Accuracy: ", attrs=['bold']), cl(round(accuracy_score(testing_b, predicted_values) * 100, 2), attrs=['bold']),
      '\n')
print(classification_report(testing_b, predicted_values, zero_division=0))
