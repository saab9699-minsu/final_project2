from django.shortcuts import render
from Test.models import News, Btc
from Test.crawling import news_crawling, upbit
import pyupbit
import mplfinance as mpf
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
        
     # 🔥 BTC 데이터 가져오기
    btc_df = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=24 * 12)

    # 🔥 데이터 전처리
    btc_df.index.name = 'Date'  # X축에 날짜 표시를 위해 인덱스명 추가

    # 🔥 이동평균선 (MA) 추가
    mav = (5, 20)  # 5분, 20분 이동평균선 추가

    # 🔥 마지막 값 강조
    last_price = btc_df['close'][-1]
    addplot = mpf.make_addplot(btc_df['close'], color='blue')

    # 🔥 차트 그리기
    fig, ax = mpf.plot(
        btc_df,
        type='candle',  # 캔들차트
        style='charles',  # 차트 스타일 (charles: 깔끔한 스타일)
        title="BTC/KRW Candlestick Chart (5-minute interval)",  # 제목
        ylabel="Price (KRW)",  # Y축 라벨
        xlabel="Time",  # X축 라벨
        mav=mav,  # 이동평균선
        volume=True,  # 거래량 표시
        addplot=addplot,  # 추가 라인
        returnfig=True  # fig 객체 반환
    )

    # 🔥 이미지로 변환
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")  # 그래프를 버퍼에 저장
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph = base64.b64encode(image_png).decode('utf-8')  # base64로 인코딩

    context = {
        "news": news,
        "graph": graph,  # 그래프를 context에 추가
        "last_price": last_price  # 마지막 가격을 추가
    }
    return render(request, "index.html", context)

def portfolio(request):
    return render(request, "portfolio.html")