from kivy.app import App
from kivy.core.window import Window
from kivy.lang.builder import Builder
from kivy.uix import dropdown
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.screenmanager import ScreenManager, Screen


class ScreenManagement(ScreenManager):
    pass


class MenuScreen(Screen):
    pass


class DropDowns(DropDown):
    pass


class FileExplorer(Screen):
    def on_released(self):
        self.add_widget(DropDowns(pos_hint={'x': 0.65, 'y': 0.4}, size_hint=(0.275, 0.3)))
    pass


class AboutUs(Screen):
    pass


class AppAbout(Screen):
    pass


sm = Builder.load_file("screen.kv")


class ScreenApp(App):
    def build(self):
        Window.clearcolor = (0.25, 0.5, 0.4, 1)
        return sm


if __name__ == '__main__':
    ScreenApp().run()
