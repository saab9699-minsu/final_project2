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
from Test.crawling import news_crawling, upbit, upbit2
import pyupbit
import mplfinance as mpf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import urllib, base64

# 포트폴리오
# 포트포리오
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices


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

    btc_df = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=24 * 12)
    btc_df.index.name = "Date"  # X축에 날짜 표시를 위해 인덱스명 추가

    # 🔥 이동평균선 (MA) 추가
    mav = (5, 20)  # 5분, 20분 이동평균선 추가

    # 🔥 최신 가격 정보
    last_price = btc_df["close"][-1]  # 최신 가격

    # 🔥 최신 가격 라인 추가
    latest_price_line = [last_price] * len(btc_df)  # 모든 행에 동일한 최신 가격 추가
    addplot = [
        mpf.make_addplot(latest_price_line, color="red", linestyle="dashed"),  # 수평선
        mpf.make_addplot(btc_df["close"], color="blue"),  # 기존의 클로즈 라인
    ]

    # 🔥 차트 그리기
    fig, ax = mpf.plot(
        btc_df,
        type="candle",  # 캔들차트
        style="charles",  # 차트 스타일
        mav=mav,  # 이동평균선
        volume=True,  # 거래량 표시
        addplot=addplot,  # 추가 라인 (수평선과 기존의 클로즈 라인)
        returnfig=True,  # fig 객체 반환
        figratio=(27, 9),  # 차트 비율 조절
    )

    # 🔥 최신 가격 텍스트 추가
    ax[0].text(
        x=len(btc_df) - 1,  # x축의 위치 (마지막 데이터 위치)
        y=last_price,  # y축의 위치 (최신 가격)
        s=f"{last_price:,.0f} KRW",  # 표시할 텍스트 (천 단위 쉼표 추가)
        color="red",  # 텍스트 색상
        fontsize=12,  # 텍스트 크기
        fontweight="bold",  # 텍스트 굵기
        verticalalignment="bottom",  # 텍스트의 세로 정렬
        horizontalalignment="left",  # 텍스트의 가로 정렬
    )

    # 🔥 X축 눈금 라벨 회전 제거
    for label in ax[0].get_xticklabels():
        label.set_rotation(0)

    # 🔥 이미지로 변환
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")  # 그래프를 버퍼에 저장
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graph = base64.b64encode(image_png).decode("utf-8")  # base64로 인코딩

    coins = upbit2()
    context = {
        "news": news,
        "graph": graph,  # 그래프를 context에 추가
        "last_price": last_price,  # 마지막 가격을 추가
        "coins": coins,
    }
    return render(request, "index.html", context)


def portfolio(request):
    # 주식, 비트코인 그래프 그리기기
    # 기본값 설정
    default_start = "2023-01-01"
    default_end = date.today().isoformat()
    default_tick = ["SPY", "GLD", "TLT"]  # 기본 종목
    default_btc = ["BTC-USD"]  # 기본 BTC 심볼
    default_price = 1000
    default_weight = [0.25, 0.25, 0.25, 0.25]

    # GET/POST 요청에서 값 가져오기
    start = request.POST.get("start", default_start)
    end = request.POST.get("end", default_end)
    price = request.POST.get("price", default_price)
    try:
        price = int(price)  # 문자열을 정수로 변환 시도
    except (ValueError, TypeError):
        # 빈 문자열 또는 변환 불가한 값일 경우 기본값 사용
        price = default_price

    # 쉼표로 구분된 입력값 처리하기
    tick_raw = request.POST.get("tick", ",".join(default_tick))
    btc_raw = request.POST.get("btc", ",".join(default_btc))

    # 리스트로 변환
    tick = [t.strip() for t in tick_raw.split(",")] if tick_raw else default_tick
    btc = [b.strip() for b in btc_raw.split(",")] if btc_raw else default_btc

    # weight 처리
    weight_raw = request.POST.get("weight", "")  # 쉼표로 구분된 weight 값
    weight = (
        [float(w.strip()) for w in weight_raw.split(",")]
        if weight_raw
        else default_weight
    )

    # 가공된 데이터 구조 생성
    set_data = {"tick": {}, "btc": {}}

    # tick과 weight 매핑
    for i, t in enumerate(tick):
        set_data["tick"][t] = weight[i] if i < len(weight) else 0.0

    # btc와 weight 매핑
    for i, b in enumerate(btc):
        btc_index = i + len(tick)  # btc의 weight는 tick 이후의 값으로 매핑
        set_data["btc"][b] = weight[btc_index] if btc_index < len(weight) else 0.0

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

    ### 포트폴리오 설정 ###
    # 실시간 환율 가져오기
    exchange_rate_data = yf.download(["USDKRW=X"], period="1d")["Adj Close"].iloc[-1]
    exchange_rate = round(exchange_rate_data.iloc[0], 2)

    # 원화 -> USD 변환
    change_price = int(price) / exchange_rate

    # 사용자 포트폴리오
    # 분산계산
    weights = np.array(weight)
    # 예상 수익률과 공분산 계산
    user_mu = expected_returns.mean_historical_return(merged_data)
    user_S = risk_models.sample_cov(merged_data)

    # 포트폴리오 설정
    user_ef = EfficientFrontier(user_mu, user_S)
    # 가중치 딕셔너리에 담기
    user_weight = {}
    for key, value in set_data.items():
        for col, weight in value.items():
            user_weight[col] = weight
    user_ef.set_weights(user_weight)
    user_port = user_ef.portfolio_performance(verbose=False)

    # 일간 수익률
    returns = merged_data.pct_change().dropna()
    # 누적 수익률
    portfolio_returns = (returns * weights).sum(axis=1)
    cumulative_returns = (1 + portfolio_returns).cumprod()

    # 비율에 따라 각 종목에 할당
    last_price = get_latest_prices(merged_data)
    user_allocation = {}
    user_leftover = change_price
    # user_buy = 0
    for key, value in set_data.items():
        for i in value:
            if key == "btc":
                # 암호화폐는 소수점 이하 단위까지 계산
                user_buy = (change_price * value[i]) / last_price[i]
                user_allocation[i] = f"{user_buy:.2f}"
                # 사용한 금액만큼 잔액 차감
                user_leftover -= user_buy * last_price[i]
            else:
                # 주식은 정수로 계산
                user_buy = (change_price * value[i]) // last_price[i]
                user_allocation[i] = int(user_buy)
                # 사용한 금액만큼 잔액 차감
                user_leftover -= user_buy * last_price[i]

    # 최적화 포트폴리오
    # 예상 수익률과 일일 자산 수익률의 연간 공분산 행렬을 계산
    mu = expected_returns.mean_historical_return(merged_data)
    S = risk_models.sample_cov(merged_data)
    # 최대 샤프 비율 최적화
    ef = EfficientFrontier(mu, S)

    # 최소 및 최대 비중 설정
    ef.add_constraint(lambda w: w >= 0.05)  # 최소 비중 5%
    ef.add_constraint(lambda w: w <= 0.7)  # 최대 비중 70%

    # 특정 자산 비중 설정 (예: TLT 최소 10%)
    ef.add_constraint(lambda w: w[merged_data.columns.get_loc("TLT")] >= 0.1)

    # 최적화 및 정리
    weights_sh = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    weights_sh = cleaned_weights  # 최적화된 가중치를 저장
    clean_weights = np.array([round(weights_sh[key], 2) for key in merged_data.columns])
    # 분산계산
    clean_weights = np.array(clean_weights)

    # 일간 수익률
    clean_returns = merged_data.pct_change().dropna()
    # 누적 수익률
    clean_portfolio_returns = (clean_returns * clean_weights).sum(axis=1)
    clean_cumulative_returns = (1 + clean_portfolio_returns).cumprod()

    # MDD
    cumulative_max = merged_data.cummax()
    drawdown = (merged_data / cumulative_max) - 1
    dd = drawdown.cummin()
    mdd = -dd.min()
    mdd_mean = round(mdd.mean(), 2) * 100

    # 포트폴리오 종목 할당 계산
    last_price = get_latest_prices(merged_data)

    allocation = {}
    leftover = change_price
    # 비율에 따라 각 종목에 할당
    for key, value in weights_sh.items():
        if key in btc:
            # 암호화폐는 소수점 이하 단위까지 계산
            btc_buy = (change_price * value) / last_price[key]
            allocation[key] = f"{btc_buy:.2f}"  # 소수점 포함
            # 사용한 금액만큼 잔액 차감
            leftover -= btc_buy * last_price[key]
        else:
            # 주식은 정수 단위로 계산
            stock_buy = (change_price * value) // last_price[key]
            allocation[key] = int(stock_buy)
            # 사용한 금액만큼 잔액 차감
            leftover -= stock_buy * last_price[key]
    # 포트폴리오 성과
    port = ef.portfolio_performance(verbose=False)

    ### 시각화 ###
    # line_grahp 생성
    line_fig = make_subplots(specs=[[{"secondary_y": True}]])

    for column in merged_data.columns:
        if column in btc:  # BTC 종목은 왼쪽 y축
            line_fig.add_trace(
                go.Scatter(
                    x=merged_data.index.tolist(),
                    y=merged_data[column].tolist(),
                    mode="lines",
                    name=column + " (BTC)",
                ),
                secondary_y=False,
            )
        else:  # 일반 종목은 오른쪽 y축
            line_fig.add_trace(
                go.Scatter(
                    x=merged_data.index.tolist(),
                    y=merged_data[column].tolist(),
                    mode="lines",
                    name=column,
                ),
                secondary_y=True,
            )

    # 레이아웃 업데이트
    line_fig.update_layout(
        width=800,
        height=500,
        xaxis_title="Date",
        yaxis=dict(title="Price (BTC)"),  # 왼쪽 y축
        yaxis2=dict(title="Price (Stocks)", overlaying="y", side="right"),  # 오른쪽 y축
    )

    # pie 그래프
    pie_fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=("사용자 설정 자산 비중", "최적화된 자산 비중"),
    )

    pie_fig.add_traces(
        go.Pie(labels=list(user_weight.keys()), values=list(user_weight.values())),
        rows=1,
        cols=1,
    )
    pie_fig.add_traces(
        go.Pie(labels=list(weights_sh.keys()), values=list(weights_sh.values())),
        rows=1,
        cols=2,
    )

    # bar 그래프
    bar_fig = make_subplots()

    col = [
        "연간 기대 수익률",
        "연간 변동성",
        "샤프 비율",
        "누적 수익률",
        "최대 낙폭(MDD)",
    ]
    user_y = [
        round(user_port[0], 2),
        round(user_port[1], 2),
        round(user_port[2], 2),
        round(cumulative_returns.iloc[-1], 2),
    ]

    optim_y = [
        round(port[0], 2),
        round(port[1], 2),
        round(port[2], 2),
        round(clean_cumulative_returns.iloc[-1], 2),
    ]

    bar_fig.add_trace(go.Bar(x=col, y=user_y, name="사용자 포트폴리오"))
    bar_fig.add_trace(go.Bar(x=col, y=optim_y, name="최적화된 포트폴리오"))

    bar_fig.update_layout(title_text="포트폴리오 성과 비교", title_x=0.5)

    # 일반 요청
    line_graph_html = line_fig.to_html(full_html=False)
    pie_graph_html = pie_fig.to_html(full_html=False)
    bar_graph_html = bar_fig.to_html(full_html=False)

    context = {
        # 설정한 포트폴리오
        "set_weight": user_weight,  # 설정한 자산 비중
        "user_allocation": user_allocation,  # 각 항목 별 개별 할당
        "user_leftover": f"{user_leftover:.2f}",
        "user_portfolio": {
            "연간 기대 수익률": f"{user_port[0]:.2f}",  # 연간 기대 수익률
            "연간 변동성": f"{user_port[1]:.2f}",  # 연간 변동성
            "샤프 비율": f"{user_port[2]:.2f}",  # 샤프비율
            "누적 수익률": round(cumulative_returns.iloc[-1], 2),  # 누적 수익률
        },
        # 최적화된 포트폴리오
        "optimized_weights": weights_sh,  # 자산 비중
        "Discrete_allocation": allocation,  # 각 항목 별 개별 할당
        "portfolio_performance": {
            "연간 기대 수익률": f"{port[0]:.2f}",  # 연간 기대 수익률
            "연간 변동성": f"{port[1]:.2f}",  # 연간 변동성
            "샤프 비율": f"{port[2]:.2f}",  # 샤프비율
            "누적 수익률": round(clean_cumulative_returns.iloc[-1], 2),  # 누적 수익률
        },
        "mdd_mean": mdd_mean,
        "Funds_remainimg": f"{leftover:.2f}",
        "exchange_rate": exchange_rate,
        # 시각화 코드
        "line_graph": line_graph_html,
        "pie_graph": pie_graph_html,
        "bar_graph": bar_graph_html,
        "default_start": start,  # 시작 날짜 유지
        "default_end": end,  # 종료 날짜 유지
        "default_tick": ",".join(tick),  # 입력한 종목 유지
        "default_btc": ",".join(btc),  # 입력한 BTC 종목 유지
    }
    return render(request, "portfolio.html", context)
