import json

from flask import Flask, request, render_template

import joblib
import pandas as pd

app = Flask(__name__, template_folder='templates')


@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        loaded_treez = joblib.load('/home/SBosq/mysite/ML/visasDT.pkl')
        fields = []
        jobTitle = request.form['jobTitle']
        ftPosition = request.form['ftPosition']
        empName = request.form['empName']
        state = request.form['state']
        empCity = request.form['empCity']
        salary = request.form['salary']
        year = request.form['year']
        fields.append(jobTitle)
        fields.append(ftPosition)
        fields.append(empName)
        fields.append(state)
        fields.append(empCity)
        fields.append(salary)
        fields.append(year)
        fields = pd.np.array(fields).reshape(1, -1)
        df = pd.DataFrame(fields)
        df.columns = ['JOB_TITLE', 'FULL_TIME_POSITION', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'WORKSITE_CITY_1',
                      'PREVAILING_WAGE_1', 'YEAR']
        dataTrain = pd.read_csv('/home/SBosq/mysite/ML/unified_Visas.csv')
        df1 = pd.DataFrame(dataTrain)
        del df1['CASE_STATUS']
        df1 = df1.append(df, ignore_index=True)
        for (columnName, columnData) in df1.iteritems():
            df1[columnName] = df1[columnName].astype('category')
            df1[columnName] = df1[columnName].cat.codes
        sample = df1.iloc[-1]
        arr_sam = sample.to_numpy().reshape(1, -1)
        y_pred = str(loaded_treez.predict(arr_sam))
        if y_pred == '[0.]':
            label_Answ = 'APPROVED'
        else:
            label_Answ = 'DENIED'
        results = label_Answ
        return render_template('results.html', value=results)
    else:
        return render_template('index.html')


@app.route('/api/visa_request', methods=['GET'])
def api_visa_request():
    if request.method == 'GET':
        field = []
        loaded_treez = joblib.load('/home/SBosq/mysite/ML/visasDT.pkl')
        # print(request.is_json)
        # content = request.get_json()
        # print(content)
        jobTitle = request.json[0]
        print(jobTitle)
        ftPosition = request.json[1]
        empName = request.json[2]
        state = request.json[3]
        empCity = request.json[4]
        salary = request.json[5]
        year = request.json[6]
        field.append(jobTitle)
        field.append(ftPosition)
        field.append(empName)
        field.append(state)
        field.append(empCity)
        field.append(salary)
        field.append(year)
        field = pd.np.array(field).reshape(1, -1)
        print(field)
        df = pd.DataFrame(field)
        df.columns = ['JOB_TITLE', 'FULL_TIME_POSITION', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'WORKSITE_CITY_1',
                      'PREVAILING_WAGE_1', 'YEAR']
        dataTrain = pd.read_csv('/home/SBosq/mysite/ML/unified_Visas.csv')
        df1 = pd.DataFrame(dataTrain)
        del df1['CASE_STATUS']
        df1 = df1.append(df, ignore_index=True)
        for (columnName, columnData) in df1.iteritems():
            df1[columnName] = df1[columnName].astype('category')
            df1[columnName] = df1[columnName].cat.codes
        sample = df1.iloc[-1]
        arr_sam = sample.to_numpy().reshape(1, -1)
        y_pred = str(loaded_treez.predict(arr_sam))
        if y_pred == '[0.]':
            label_Answ = 'APPROVED'
        else:
            label_Answ = 'DENIED'
        results = json.dumps(label_Answ)
        print(results)
        return results


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host="127.0.0.1", port=80)
