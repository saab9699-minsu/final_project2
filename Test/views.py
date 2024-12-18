from django.shortcuts import render
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from django.http import JsonResponse

# plotly 템플릿
import plotly.io as pio

pio.templates.default = "plotly_white"

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

# 디테일 페이지 
# 반감기 패턴 페이지 
def halving_pattern(request):
    return render(request, "detail_halving_pattern.html")


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
    btc_df.index.name = "Date"  # X축에 날짜 표시를 위해 인덱스명 추가

    # 🔥 이동평균선 (MA) 추가
    mav = (5, 20)  # 5분, 20분 이동평균선 추가

    # 🔥 마지막 값 강조
    last_price = btc_df["close"][-1]
    addplot = mpf.make_addplot(btc_df["close"], color="blue")

    # 🔥 차트 그리기
    fig, ax = mpf.plot(
        btc_df,
        type="candle",  # 캔들차트
        style="charles",  # 차트 스타일 (charles: 깔끔한 스타일)
        title="BTC/KRW Candlestick Chart (5-minute interval)",  # 제목
        ylabel="Price (KRW)",  # Y축 라벨
        xlabel="Time",  # X축 라벨
        mav=mav,  # 이동평균선
        volume=True,  # 거래량 표시
        addplot=addplot,  # 추가 라인
        returnfig=True,  # fig 객체 반환
    )

    # 🔥 이미지로 변환
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")  # 그래프를 버퍼에 저장
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph = base64.b64encode(image_png).decode("utf-8")  # base64로 인코딩

    context = {
        "news": news,
        "graph": graph,  # 그래프를 context에 추가
        "last_price": last_price,  # 마지막 가격을 추가
    }
    return render(request, "index.html", context)


def portfolio(request):
    # 주식, 비트코인 그래프 그리기기
    # 기본값 설정
    default_start = "2023-01-01"
    default_end = date.today().isoformat()
    default_tick = ["SPY", "GLD", "TLT"]  # 기본 종목
    default_btc = ["BTC-USD"]  # 기본 BTC 심볼

    # GET/POST 요청에서 값 가져오기
    start = request.POST.get("start", default_start)
    end = request.POST.get("end", default_end)
    tick = request.POST.getlist("tick") or default_tick
    btc = request.POST.getlist("btc") or default_btc

    # tick과 btc를 리스트로 변환
    tick_raw = request.POST.get("tick", ",".join(default_tick))  # 쉼표로 구분된 기본값
    btc_raw = request.POST.get("btc", ",".join(default_btc))  # 쉼표 구분된 기본값

    tick = (
        [t.strip() for t in tick_raw.split(",")]
        if isinstance(tick_raw, str)
        else tick_raw
    )
    btc = (
        [b.strip() for b in btc_raw.split(",")] if isinstance(btc_raw, str) else btc_raw
    )

    # 데이터 가져오기
    try:
        usd = yf.download(tick, start=start, end=end)["Adj Close"]
        btc_data = yf.download(btc, start=start, end=end)["Adj Close"]

        # 데이터 병합
        merged_data = (
            pd.merge(
                btc_data,
                usd,
                how="outer",
                left_on=btc_data.index,
                right_on=usd.index,
            )
            .ffill()
            .bfill()
            .set_index("key_0")
        )
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=400)

    # plotly 그래프 생성성
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    for column in merged_data.columns:
        if column in btc:  # BTC 종목은 왼쪽 y축
            fig.add_trace(
                go.Scatter(
                    x=merged_data.index.tolist(),
                    y=merged_data[column].tolist(),
                    mode="lines",
                    name=column + " (BTC)",
                ),
                secondary_y=False,
            )
        else:  # 일반 종목은 오른쪽 y축
            fig.add_trace(
                go.Scatter(
                    x=merged_data.index.tolist(),
                    y=merged_data[column].tolist(),
                    mode="lines",
                    name=column,
                ),
                secondary_y=True,
            )

    # 레이아웃 업데이트
    fig.update_layout(
        width=800,
        height=500,
        xaxis_title="Date",
        yaxis=dict(title="Price (BTC)"),  # 왼쪽 y축
        yaxis2=dict(title="Price (Stocks)", overlaying="y", side="right"),  # 오른쪽 y축
    )

    if request.headers.get("x-requested-with") == "XMLHttpRequest":
        graph_html = fig.to_html(full_html=False)  # Plotly 그래프를 HTML로 변환
        return JsonResponse({"graph_html": graph_html}, safe=False)  # HTML 반환

    # 일반 요청
    graph_html = fig.to_html(full_html=False)

    context = {
        "graph": graph_html,
        "default_start": start,  # 시작 날짜 유지
        "default_end": end,  # 종료 날짜 유지
        "default_tick": ",".join(tick),  # 입력한 종목 유지
        "default_btc": ",".join(btc),  # 입력한 BTC 종목 유지
    }
    return render(request, "portfolio.html", context)
