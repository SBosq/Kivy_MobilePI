# -- coding: utf-8 --
from kivymd.app import MDApp
from kivy.metrics import dp
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.screenmanager import Screen, ScreenManager
import requests
import json


class MenuScreen(Screen):
    pass


class AboutScreen(Screen):
    pass


class DevsScreen(Screen):
    pass


class ProbScreen(Screen):
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


sm = ScreenManager()
sm.add_widget(MenuScreen(name='menu'))
sm.add_widget(AboutScreen(name='about'))
sm.add_widget(DevsScreen(name='devs'))
sm.add_widget(ProbScreen(name='prob'))


class MLApp(MDApp):

    def build(self):
        self.theme_cls.primary_palette = "Blue"
        self.theme_cls.primary_hue = "800"
        screen = Builder.load_file("screen.kv")

        return screen


MLApp().run()
