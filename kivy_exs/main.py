from kivy.app import App
from kivy.metrics import dp
from kivy.core.window import Window
from kivy.lang.builder import Builder
from kivy.uix.dropdown import DropDown
from kivy.uix.screenmanager import ScreenManager, Screen

import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import DataConversionWarning
import joblib

warnings.filterwarnings(action='ignore', category=DataConversionWarning)


class ScreenManagement(ScreenManager):
    pass


class MenuScreen(Screen):
    pass


class DropDowns(DropDown):
    pass


answers = []


class CalcProb(Screen):
    def limit_spinner(self, *args):
        max = 5  # max number of Buttons to display in the Spinner DropDown
        self.spinner.dropdown_cls.max_height = max * dp(
            48)  # dp(48) is the size of each Button in the DropDown (from style.kv)

    def calculate(self):
        loaded_treez = joblib.load('ML/visasDT.pkl')
        fields = []
        answ1 = self.ids.jobTitle.text
        fields.append(answ1)
        answ2 = self.ids.ftPosition.text
        fields.append(answ2)
        answ3 = self.ids.empName.text
        fields.append(answ3)
        answ4 = self.ids.state.text
        fields.append(answ4)
        answ5 = self.ids.empCity.text
        fields.append(answ5)
        answ6 = self.ids.salary.text
        fields.append(answ6)
        answ7 = self.ids.year.text
        fields.append(answ7)
        fields = np.array(fields).reshape(1, -1)
        df = pd.DataFrame(fields)
        df.columns = ['JOB_TITLE', 'FULL_TIME_POSITION', 'EMPLOYER_NAME', 'EMPLOYER_STATE', 'WORKSITE_CITY_1',
                      'PREVAILING_WAGE_1', 'YEAR']
        dataTrain = pd.read_csv('ML/unified_Visas.csv')
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
        answers.append(label_Answ)
        self.ids.result_ML.text = str(label_Answ)
    pass


class AboutUs(Screen):
    pass


class AppAbout(Screen):
    pass


sm = Builder.load_file("screen.kv")


class ScreenApp(App):
    def build(self):
        Window.clearcolor = (224/255, 224/255, 224/255, 224/255)
        return sm


if __name__ == '__main__':
    ScreenApp().run()
