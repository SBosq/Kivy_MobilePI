
# -- coding: utf-8 --
from kivy.app import App
from kivy.metrics import dp
from kivy.lang.builder import Builder
from kivy.properties import StringProperty
from kivy.uix.dropdown import DropDown
from kivy.uix.screenmanager import ScreenManager, Screen
import requests
import json


class ScreenManagement(ScreenManager):
    pass


class MenuScreen(Screen):
    pass


class DropDowns(DropDown):
    pass


answers = []


class CalcProb(Screen):
    def limit_spinner(self, *args):
        max_items = 5  # max number of Buttons to display in the Spinner DropDown
        self.spinner.dropdown_cls.max_height = max_items * dp(
            48)  # dp(48) is the size of each Button in the DropDown (from style.kv)

    def calculate(self):
        position = StringProperty("")
        fields = {}
        answ1 = self.ids.jobTitle.text
        fields["jobTitle"] = answ1
        answ2 = self.ids.ftPosition.text
        fields["ftPosition"] = answ2
        answ3 = self.ids.empName.text
        fields["empName"] = answ3
        answ4 = self.ids.state.text
        fields["state"] = answ4
        answ5 = self.ids.empCity.text
        fields["empCity"] = answ5
        answ6 = self.ids.salary.text
        fields["salary"] = answ6
        answ7 = self.ids.year.text
        fields["year"] = answ7
        params = json.dumps(fields, indent=4)
        request = requests.get('http://sbosq.pythonanywhere.com/api/visa_request', json=params)
        self.ids.result_ML.text = request.json()
        request.close()

    pass


class AboutUs(Screen):
    pass


class AppAbout(Screen):
    pass


sm = Builder.load_file("screen.kv")


class ScreenApp(App):
    def build(self):
        return sm


if __name__ == '__main__':
    ScreenApp().run()
