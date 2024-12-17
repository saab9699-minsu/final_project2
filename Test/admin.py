from django.contrib import admin
from Test.models import News, Btc

# Register your models here.

@admin.register(News)
class NewsAdmin(admin.ModelAdmin):
    list_display = [
        "title",
        "company",
        "date",
        "href"
    ]

@admin.register(Btc)
class NewsAdmin(admin.ModelAdmin):
    list_display = [
        "close",
        "open",
        "low",
        "high"
    ]

@admin.register(News)
class NewsAdmin(admin.ModelAdmin):
    list_display = [
        "title",
        "company",
        "date",
        "href"
    ]

@admin.register(Btc)
class NewsAdmin(admin.ModelAdmin):
    list_display = [
        "close",
        "open",
        "low",
        "high"
    ]