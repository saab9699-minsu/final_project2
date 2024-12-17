from django.urls import path
from Test import views

app_name = "test"

urlpatterns = [
    path("about/", views.about, name="about"),
    path("contact/", views.contact, name="contact"),
    path("do/", views.do, name="do"),
    path("index/", views.index, name="index"),
    path("portfolio/", views.portfolio, name="portfolio"),
]