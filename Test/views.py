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


# Create your views here.


def about(request):
    return render(request, "about.html")


def contact(request):
    return render(request, "contact.html")


def do(request):
    return render(request, "do.html")


def index(request):
    return render(request, "index.html")


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
