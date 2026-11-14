from django.urls import path
from .views import home, login_view,register_view,dashboard,live_recognition

urlpatterns = [
    path("", home, name="home"),
    path("login/", login_view, name="login"),
    path("register/",register_view, name="register"),
    path("dashboard/",dashboard, name="dashboard"),
    path("live/",live_recognition, name="live"),
]