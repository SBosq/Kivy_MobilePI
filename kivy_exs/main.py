from kivy.app import App
from kivy.core.window import Window
from kivy.lang.builder import Builder
from kivy.uix.screenmanager import ScreenManager, Screen


class ScreenManagement(ScreenManager):
    pass


class MenuScreen(Screen):
    pass


class FileExplorer(Screen):
    pass


class AboutUs(Screen):
    pass


class AppAbout(Screen):
    pass


sm = Builder.load_file("screen.kv")


class ScreenApp(App):
    def build(self):
        # Window.clearcolor = (2, 3, 8, 1)
        return sm


if __name__ == '__main__':
    ScreenApp().run()
