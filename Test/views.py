from django.shortcuts import render
from Test.models import News, Btc
from Test.crawling import news_crawling, upbit
import pyupbit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import urllib, base64


# Create your views here.

def about(request):
    return render(request, "about.html")

def contact(request):
    return render(request, "contact.html")

def do(request):
    return render(request, "do.html")

def index(request):
    news = News.objects.all()
    news.delete()
    df = news_crawling()
    for i in range(len(df)):
        News.objects.create(
            title=df.iloc[i]["title"],
            company=df.iloc[i]["company"],
            date=df.iloc[i]["date"],
            href=df.iloc[i]["href"],
        )
        
    btc_df = pyupbit.get_ohlcv("KRW-BTC", interval = "minute5", count = 24 * 12)
    plt.figure(figsize=(10, 5))
    plt.plot(btc_df.index, btc_df['close'], label="Close Price", color='blue')
    plt.xlabel('Time')
    plt.ylabel('Price (KRW)')
    plt.title('BTC 5-minute Interval Price')
    plt.legend()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph = base64.b64encode(image_png).decode('utf-8')  # base64로 인코딩
    plt.close()

    context = {
        "news" : news,
        "graph" : graph
    }
    return render(request, "index.html", context)

def portfolio(request):
    return render(request, "portfolio.html")