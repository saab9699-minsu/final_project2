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

    path("detail_halving_pattern/", views.halving_pattern, name="halving_pattern"),
    path("detail_issue/", views.detail_issue, name="detail_issue"),
]