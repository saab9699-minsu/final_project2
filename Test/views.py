from django.shortcuts import render
from Test.models import News, Btc
from Test.crawling import news_crawling, upbit
import pyupbit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import urllib, base64
from mpl_finance import candlestick2_ochl


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

    fig = plt.figure(figsize=(15, 5))  # 차트의 크기 설정
    ax1 = fig.add_subplot(1, 1, 1)  # 1행 1열의 서브플롯
    candlestick2_ochl(
        ax1, 
        btc_df['open'], btc_df['close'], btc_df['high'], btc_df['low'], 
        width=0.7, 
        colorup='r', colordown='b'
    )
    plt.xlabel('Time')
    plt.ylabel('Price (KRW)')
    plt.title('BTC Candlestick Chart (5-minute interval)')

    # 🔥 그래프를 이미지로 변환
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph = base64.b64encode(image_png).decode('utf-8')  # base64로 인코딩
    plt.close()  # 🔥 plt.close()로 리소스 해제

    context = {
        "news": news,
        "graph": graph  # 그래프 이미지를 컨텍스트에 추가
    }
    return render(request, "index.html", context)

def portfolio(request):
    return render(request, "portfolio.html")