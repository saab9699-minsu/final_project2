from django.db import models

# Create your models here.
class News(models.Model):
    title = models.TextField("제목", blank = True)
    company = models.CharField("참조사이트", max_length=50)
    date = models.CharField("날짜", max_length=50)
    href = models.TextField("주소", blank = True)

class Btc(models.Model):
    close = models.IntegerField("종가", blank=True)
    open = models.IntegerField("시가", blank=True)
    low = models.IntegerField("저가", blank=True)
    high = models.IntegerField("고가", blank=True)