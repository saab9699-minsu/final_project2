<<<<<<< HEAD
from django.urls import path
from Test import views

app_name = "test"

urlpatterns = [
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
    path("do/", views.do, name="do"),
    path("index/", views.index, name="index"),
    path("portfolio/", views.portfolio, name="portfolio"),
=======
from django.urls import path
from Test import views

app_name = "test"

urlpatterns = [
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
    path("do/", views.do, name="do"),
    path("index/", views.index, name="index"),
    path("", views.index, name="home"),
    path("portfolio/", views.portfolio, name="portfolio"),
>>>>>>> 15ddfc0ab029985e33c5f3b7aeba4753d9837705
]