from django.shortcuts import render
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
from datetime import datetime, timedelta
from django.http import JsonResponse
from django.conf import settings
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
import time

from sklearn.preprocessing import MinMaxScaler

# plotly 템플릿
import plotly.io as pio
import plotly.express as px

pio.templates.default = "plotly_white"
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# 긍부정분류
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from Test.crawling import news_crawling, upbit, upbit2
import pyupbit
import mplfinance as mpf
from mpl_finance import candlestick2_ochl
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
from keras.models import load_model

# 포트폴리오
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# 캐쉬 저장
from django.core.cache import cache

# Create your views here.


def about(request):
    return render(request, "about.html")


# ===================
# ====== model ======
# ===================
def contact(request):
    df = upbit()
    fig = plt.figure(figsize=(5, 5))  # 224x224에 맞추기 위해 5x5 인치로 설정
    ax1 = fig.add_subplot(1, 1, 1)

    candlestick2_ochl(
        ax1,
        df["open"],
        df["close"],
        df["high"],
        df["low"],
        width=1,
        colorup="r",
        colordown="b",
    )

    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.axis("off")
    plt.tight_layout()

    # 이미지를 버퍼에 저장 (PNG 형식)**
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    buffer.seek(0)
    image_png = buffer.getvalue()
    cnn_chart = base64.b64encode(image_png).decode("utf-8")

    # PIL로 이미지를 열고 크기를 조정
    image = Image.open(buffer)
    image = image.convert("RGB")
    image = image.resize((224, 224))  # CNN 모델 입력 크기 (224x224)로 변경
    image_array = np.array(image)  # Numpy 배열로 변환 (224, 224, 3)

    buffer.close()
    plt.close(fig)

    model = load_model("./model/cnn/CNN-20-224-3(0.54).keras")

    image_array = image_array / 255.0  # 픽셀 값을 0~1 사이로 정규화
    image_array = np.expand_dims(
        image_array, axis=0
    )  # (224, 224, 3) -> (1, 224, 224, 3)

    # 모델에 예측 수행
    prediction = model.predict(image_array)

    up_down = np.argmax(prediction, axis=1)
    per = (prediction[0].max() * 100).round(3)
    up_down2 = "상승"
    if up_down[0] == 0:
        up_down2 = "하락"

    # 오늘 날짜
    today = pd.to_datetime(datetime.today().strftime("%Y-%m-%d"))

    # 2024년 12월 20일까지 업데이트 된 파일
    train = pd.read_csv("./data/web_btc.csv", index_col=0)
    last_updatedate = pd.to_datetime(train["ds"].iloc[-1]) + timedelta(1)

    # 최근 날짜 데이터 업비트에서 불러오기
    btc_df = pyupbit.get_ohlcv("USDT-BTC", interval="day", count=10)
    btc_df = btc_df.reset_index()
    btc_df = btc_df.loc[btc_df["index"] > last_updatedate, ["index", "close"]]
    btc_df.columns = ["ds", "y"]
    btc_df["ds"] = btc_df["ds"].dt.strftime("%Y-%m-%d")

    # 최근 자료 결합 및 저장
    train = pd.concat([train, btc_df], axis=0).reset_index(drop=True)
    train.to_csv("./data/web_btc.csv")

    # 프로펫 예측 데이터 불러오기
    fcst = pd.read_csv("./data/web_fcst.csv", index_col=0)

    # 프로펫 그래프 그리기
    # Figure 생성하기
    fig = go.Figure()

    # 아래쪽 경계 (yhat_lower)
    fig.add_trace(
        go.Scatter(
            x=fcst["ds"],
            y=fcst["yhat_lower"],
            fill=None,
            line_color="rgba(0,100,80,0)",  # 경계선을 숨김
            name="예측값 범위(하단)",  # 범례 이름
            showlegend=True,
        )
    )

    # 위쪽 경계 (yhat_upper), 아래쪽과의 영역을 채움
    fig.add_trace(
        go.Scatter(
            x=fcst["ds"],
            y=fcst["yhat_upper"],
            fill="tonexty",  # 이전 y 값과의 영역을 채움
            fillcolor="skyblue",  # 채움 색상
            line_color="rgba(0,100,80,0)",  # 경계선을 숨김
            name="예측값 범위(상단)",  # 범례 이름
            showlegend=True,
        )
    )

    # 예측 값 표시
    fig.add_trace(
        go.Scatter(
            x=fcst["ds"],
            y=fcst["yhat"],
            line_color="#1f77b4",
            name="예측값",  # 범례 이름
            showlegend=True,
        )
    )

    # 실제값 표시
    fig.add_trace(
        go.Scatter(
            x=train.loc[train["ds"] > "2016-07-09", "ds"],
            y=train.loc[train["ds"] > "2016-07-09", "y"],
            name="실제값",  # 범례 이름
            showlegend=True,
        )
    )

    # 슬라이더 추가 (3개월 단위)
    fig.update_layout(
        xaxis=dict(
            rangeslider=dict(visible=True),  # 슬라이더 활성화
            # (12개월)1달 이후 포함
            range=[today - pd.DateOffset(months=11), today + pd.DateOffset(months=1)],
        ),
        template="plotly_white",
        width=1100,
        height=600,
    )

    fig.update_traces(mode="lines")

    prophet = fig.to_html(full_html=False)

    # 예측 테이블
    # 날짜 업데이트해서 장기 예측
    time_li = [30, 60, 90, 180, 365]
    date_later_li = []
    pred_date_li = []
    pred_btc_li = []

    for i in range(len(time_li)):
        date_later = time_li[i]
        pred_date = (
            fcst.loc[
                fcst["ds"] == (today + timedelta(time_li[i])).strftime("%Y-%m-%d"), "ds"
            ]
            .astype("str")
            .values[0]
        )
        pred_btc = round(
            fcst.loc[
                fcst["ds"] == (today + timedelta(time_li[i])).strftime("%Y-%m-%d"),
                "yhat",
            ].values[0],
            -3,
        )

        date_later_li.append(f"{date_later}일 후")
        pred_date_li.append(pred_date)
        pred_btc_li.append(f"약 {pred_btc} $")

    df = pd.DataFrame(
        {"구분": date_later_li, "예측일": pred_date_li, "예측가격": pred_btc_li}
    )

    df = df.set_index("구분").T
    df.reset_index(inplace=True)

    fig_table = ff.create_table(df)

    fig_table.update_layout(template="plotly_white", width=1130, height=200)

    prophet_table = fig_table.to_html(full_html=False)

    context = {
        # CNN 예측 모델
        "cnn_chart": cnn_chart,
        "prediction": prediction,
        "per": per,
        "up_down2": up_down2,
        # prophet 예측 모델
        "prophet": prophet,
        "prophet_table": prophet_table,
    }

    return render(request, "contact.html", context)


def do(request):
    return render(request, "do.html")


# 디테일 페이지
# 반감기 패턴 페이지
def halving_pattern(request):
    # 누적수익률
    detail_df = pd.read_csv("./data/log_profit_btc.csv")
    fig = px.line(
        detail_df,
        x="index",
        y="누적수익률",
        color="반감기",
        labels=dict(index="반감기 이후 개월수"),
    )

    fig.update_layout(
        width=700, height=400, title_text="<b>누적수익률_반감기</b>", title_x=0.5
    )

    log_profit_btc_graph = fig.to_html(full_html=False)

    # 누적, 구간 그래프
    hl_1_3 = detail_df.loc[detail_df["반감기"].isin(["1차", "2차", "3차"])]
    fig_2 = px.line(
        hl_1_3, x="index", y="log누적수익률", color="구분", facet_col="반감기"
    )
    fig_2.update_xaxes(title="개월")
    fig_2.update_layout(width=800, height=300)

    # HTML 파일로 저장
    log_profit_btc_category = fig_2.to_html(full_html=False)

    context = {
        # 누적수익률 그래프
        "log_profit_btc_graph": log_profit_btc_graph,
        # 누적, 구간 log 수익률 그래프
        "log_profit_btc_category": log_profit_btc_category,
    }

    return render(request, "detail_halving_pattern.html", context)


def detail_issue(request):
    return render(request, "detail_issue.html")


def technical_analysis(request):
    return render(request, "detail_technical_analysis.html")


# ==================
# ====== main ======
# ==================
def index(request):

    # ==========================
    # ======= 뉴스 실시간 =======
    # ==========================
    news = cache.get("news_cache")
    if news is None:
        news = news_crawling()[:5]
        cache.set("news_cache", news, 60 * 10)

    pie_chart = cache.get("pie_chart_cache")
    if pie_chart is None:
        tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
        model = AutoModelForSequenceClassification.from_pretrained(
            "snunlp/KR-FinBert-SC"
        )

        classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

        news2 = news_crawling()
        news2 = pd.DataFrame(news2)

        label_list1 = []
        for i in range(len(news2)):
            label1, score1 = classifier(news2.iloc[i, 0])[0].values()
            label_list1.append(label1)

        news2["분류"] = label_list1

        news2["하락"] = news2["title"].map(lambda x: 1 if x.find("하락") > 0 else 0)
        news2["상승"] = news2["title"].map(lambda x: 1 if x.find("상승") > 0 else 0)
        news2.loc[(news2["하락"] == 1) & (news2["분류"] != "positive"), "분류"] = (
            "negative"
        )
        news2.loc[(news2["상승"] == 1) & (news2["분류"] != "negative"), "분류"] = (
            "positive"
        )

        news2_counts = news2["분류"].value_counts()
        ratio = news2_counts.loc[["neutral", "positive", "negative"]].values
        labels = news2_counts.loc[["neutral", "positive", "negative"]].index
        explode = [0.05, 0.05, 0.05]
        plt.figure(figsize=(4, 3))
        plt.pie(
            ratio,
            labels=labels,
            autopct="%.1f%%",
            startangle=260,
            counterclock=False,
            explode=explode,
            shadow=True,
            colors=["gray", "skyblue", "coral"],
        )
        plt.axis("equal")
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        pie_chart = base64.b64encode(image_png).decode("utf-8")
        plt.close()
        cache.set("pie_chart_cache", pie_chart, 60 * 10)

    # ================================
    # ===== 비트코인 실시간 그래프 =====
    # ================================

    graph = cache.get("graph_cache")
    if graph is None:
        btc_df = pyupbit.get_ohlcv("KRW-BTC", interval="minute5", count=24 * 12)
        btc_df.index.name = "Date"  # X축에 날짜 표시를 위해 인덱스명 추가

        # 이동평균선 설정
        mav = (5, 20)  # 5분, 20분 이동평균선 추가

        # 최신 가격 정보
        last_price = btc_df["close"].iloc[-1]  # 최신 가격

        # 빨간색 점선 가로선 추가
        latest_price_line = [last_price] * len(
            btc_df
        )  # 모든 행에 동일한 최신 가격 추가
        addplot = [
            mpf.make_addplot(
                latest_price_line, color="red", linestyle="dashed", alpha=0.5
            ),  # 빨간색 점선 가로선
        ]

        # 차트 스타일 설정 (음봉 파란색, 양봉 빨간색)
        mc = mpf.make_marketcolors(
            up="red",  # 양봉 (상승) 색상
            down="blue",  # 음봉 (하락) 색상
            edge="inherit",  # 테두리 색상은 캔들 색상과 동일
            wick="inherit",  # 심지 색상은 캔들 색상과 동일
            volume="in",  # 거래량 색상 설정
        )
        s = mpf.make_mpf_style(marketcolors=mc, gridcolor="#e6e6e6")  # 스타일 설정

        # 차트 그리기
        fig, ax = mpf.plot(
            btc_df,
            type="candle",  # 캔들 차트
            style=s,  # 스타일 적용
            mav=mav,  # 이동평균선
            volume=True,  # 거래량 표시
            addplot=addplot,  # 빨간색 점선 가로선 추가
            returnfig=True,  # fig 객체 반환
            figratio=(27, 9),  # 차트 비율 조절
        )

        # 최신 가격 텍스트 추가
        ax[0].text(
            x=len(btc_df) + 30,  # x축 위치 (마지막 데이터 위치)
            y=last_price,  # y축 위치 (최신 가격)
            s=f"{last_price:,.0f} KRW",  # 천 단위 쉼표 추가
            color="white",  # 텍스트 색상
            fontsize=12,  # 폰트 크기
            fontweight="bold",  # 폰트 굵기
            verticalalignment="center",  # 세로 정렬
            horizontalalignment="center",  # 가로 정렬
            bbox=dict(
                facecolor="red",  # 박스 배경 색상
                edgecolor="none",  # 테두리 제거
                boxstyle="larrow,pad=0.4",  # 왼쪽 화살표 모양
                alpha=0.9,  # 투명도 (0=투명, 1=불투명)
            ),
        )

        # 이미지로 변환
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png")  # 그래프를 버퍼에 저장
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        graph = base64.b64encode(image_png).decode("utf-8")  # base64로 인코딩
        plt.close()
        cache.set("graph_cache", graph, 60)

    # ==========================
    # ===== 코인시세 실시간 =====
    # ==========================

    coins = cache.get("coins_cache")
    if coins is None:
        try:
            coins = upbit2()
        except:
            coins = coins2
        cache.set("coins_cache", coins, 10)
    else:
        coins2 = coins

    context = {
        "graph": graph,  # 실시간 비트코인시세 그래프
        "coins": coins,  # 실시간 코인시세
        "news": news,  # 실시간 뉴스
        "pie_chart": pie_chart,  # 실시간 긍부정 pie차트
    }
    return render(request, "index.html", context)


def portfolio(request):
    # 주식, 비트코인 그래프 그리기기
    # 기본값 설정
    default_start = "2024-01-01"  # 시작 날짜 기본값
    default_end = date.today().isoformat()  # 종료날짜 기본값
    default_tick = ["GLD", "SPY", "TLT"]  # 기본 종목
    default_btc = ["BTC-USD"]  # 기본 BTC 심볼
    default_price = 5000000
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

    # 쉼표로 구분된 weight 값
    error_message = None
    weight_raw = request.POST.get("weight", "")
    try:
        # 리스트로 변환환
        weight = (
            [float(w.strip()) for w in weight_raw.split(",")]
            if weight_raw
            else default_weight
        )
        if sum(weight) != 1:
            error_message = "합이 1이 아닙니다. 기본 가중치를 사용합니다."
            weight = default_weight
    except (ValueError, TypeError):
        error_message = "유효하지 않은 입력입니다. 기본 가중치를 사용합니다."
        weight = default_weight

    # 가공된 데이터 구조 생성
    set_data = {"btc": {}, "tick": {}}

    # tick과 weight 매핑
    for i, t in enumerate(tick):
        set_data["tick"][t] = weight[i] if i < len(weight) else 0.0

    # btc와 weight 매핑
    for i, b in enumerate(btc):
        btc_index = i + len(tick)  # btc의 weight는 tick 이후의 값으로 매핑
        set_data["btc"][b] = weight[btc_index] if btc_index < len(weight) else 0.0

    # 데이터 가져오기
    try:
        usd = yf.download(tick, start=start, end=end, progress=False)["Adj Close"]
        btc_data = yf.download(btc, start=start, end=end, progress=False)["Adj Close"]

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

    default_usd = yf.download(default_tick, start=start, end=end, progress=False)[
        "Adj Close"
    ]
    default_btc_data = yf.download(default_btc, start=start, end=end, progress=False)[
        "Adj Close"
    ]

    default_df = (
        pd.merge(
            default_btc_data,
            default_usd,
            how="outer",
            left_on=default_btc_data.index,
            right_on=default_usd.index,
        )
        .ffill()
        .bfill()
        .set_index("key_0")
    )

    # ==========================
    # ===== 포트폴리오 설정 =====
    # ==========================
    # 실시간 환율 가져오기
    exchange_rate_data = yf.download(["USDKRW=X"], period="1d", progress=False)[
        "Adj Close"
    ].iloc[-1]
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
    for key, value in user_weight.items():
        #  for i in value:
        if key in btc:
            # 암호화폐는 소수점 이하 단위까지 계산
            user_buy = (change_price * value) / last_price[key]
            btc_value_in_dollars = user_buy * last_price[key]
            user_allocation[key] = round(btc_value_in_dollars, 2)
            # 사용한 금액만큼 잔액 차감
            user_leftover -= user_buy * last_price[key]
        else:
            # 주식은 정수로 계산
            user_buy = (change_price * value) // last_price[key]
            user_allocation[key] = int(user_buy)
            # 사용한 금액만큼 잔액 차감
            user_leftover -= user_buy * last_price[key]

    # 고정된 포트폴리오
    # 예상 수익률과 일일 자산 수익률의 연간 공분산 행렬을 계산
    mu = expected_returns.mean_historical_return(default_df)
    S = risk_models.sample_cov(default_df)
    # 최대 샤프 비율 최적화
    ef = EfficientFrontier(mu, S)

    weight_dict = {}
    for i in default_df:
        for j in default_weight:
            weight_dict[i] = j

    # 지정한 weught
    ef.set_weights(weight_dict)

    # 분산계산
    fix_weights = np.array(weight_dict)

    # 일간 수익률
    fix_returns = default_df.pct_change().dropna()
    # 누적 수익률
    fix_portfolio_returns = (fix_returns * fix_weights).sum(axis=1)
    fix_cumulative_returns = (1 + fix_portfolio_returns).cumprod()

    allocation = {}
    leftover = change_price
    # 비율에 따라 각 종목에 할당
    for key, value in weight_dict.items():
        if key in btc:
            # 암호화폐는 소수점 이하 단위까지 계산
            btc_buy = (change_price * value) / last_price[key]
            btc_value_in_dollars = btc_buy * last_price[key]
            allocation[key] = round(btc_value_in_dollars, 2)  # 소수점 포함
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

    # 사용자 포트폴리오 MDD 계산
    # 각 자산의 수익률에 비중을 곱하고, 각 날짜벼로 모든 자산의 수익률을 합산
    user_portfolio_returns = (returns * weights).sum(axis=1)
    user_cumulative_value = (1 + user_portfolio_returns).cumprod()  # 누적가치치
    user_cumulative_max = user_cumulative_value.cummax()  # 누적 최고점
    user_drawdown = ((user_cumulative_value / user_cumulative_max) - 1) * 100  # 하락률
    user_mdd = -user_drawdown.min()  # 최대 하락률

    # 고정된 포트폴리오 MDD 계산
    # 고정된 포트폴리오의 일별 수익률 / 각자산의 비중에 모든 자산의 수익률을 합산
    fixed_daily_returns = (fix_returns * fix_weights).sum(
        axis=1
    )  # 고정 포트폴리오 가치
    fixed_cumulative_value = (1 + fixed_daily_returns).cumprod()  # 누적가치
    fixed_cumulative_max = fixed_cumulative_value.cummax()  # 누적 최고점
    fixed_drawdown = (
        (fixed_cumulative_value / fixed_cumulative_max) - 1
    ) * 100  # 하락률
    fixed_mdd = -fixed_drawdown.min()  # 최대 하락률

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
        subplot_titles=("사용자 설정 자산 비중", "고정된 자산 비중"),
    )

    pie_fig.add_traces(
        go.Pie(labels=list(user_weight.keys()), values=list(user_weight.values())),
        rows=1,
        cols=1,
    )
    pie_fig.add_traces(
        go.Pie(labels=list(weight_dict.keys()), values=list(weight_dict.values())),
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
        round(fix_cumulative_returns.iloc[-1], 2),
    ]

    bar_fig.add_trace(go.Bar(x=col, y=user_y, name="사용자 포트폴리오"))
    bar_fig.add_trace(go.Bar(x=col, y=optim_y, name="고정된 포트폴리오"))

    bar_fig.update_layout(title_text="포트폴리오 성과 비교", title_x=0.5)

    # 일반 요청
    line_graph_html = line_fig.to_html(full_html=False)
    pie_graph_html = pie_fig.to_html(full_html=False)
    bar_graph_html = bar_fig.to_html(full_html=False)

    context = {
        # 설정한 포트폴리오
        "set_weight": user_weight,  # 설정한 자산 비중
        "user_allocation": user_allocation,  # 각 항목 별 개별 할당
        "user_leftover": f"{user_leftover:.2f}",  # 남은자금
        "user_portfolio": {
            "연간 기대 수익률": f"{user_port[0]:.2f}",  # 연간 기대 수익률
            "연간 변동성": f"{user_port[1]:.2f}",  # 연간 변동성
            "샤프 비율": f"{user_port[2]:.2f}",  # 샤프비율
            "누적 수익률": round(cumulative_returns.iloc[-1], 2),  # 누적 수익률
        },
        "user_mdd": round(user_mdd, 2),
        # 고정된 포트폴리오
        "optimized_weights": weight_dict,  # 자산 비중
        "Discrete_allocation": allocation,  # 각 항목 별 개별 할당
        "leftover": f"{leftover:.2f}",  # 남은 자금
        "portfolio_performance": {
            "연간 기대 수익률": f"{port[0]:.2f}",  # 연간 기대 수익률
            "연간 변동성": f"{port[1]:.2f}",  # 연간 변동성
            "샤프 비율": f"{port[2]:.2f}",  # 샤프비율
            "누적 수익률": round(fix_cumulative_returns.iloc[-1], 2),  # 누적 수익률
        },
        "fix_mdd": round(fixed_mdd, 2),
        # "mdd_mean": mdd_mean,  # MDD
        "exchange_rate": exchange_rate,  # 환율
        # 시각화 코드
        "line_graph": line_graph_html,
        "pie_graph": pie_graph_html,
        "bar_graph": bar_graph_html,
        "default_start": start,  # 시작 날짜 유지
        "default_end": end,  # 종료 날짜 유지
        "default_tick": ",".join(tick),  # 입력한 종목 유지
        "default_btc": ",".join(btc),  # 입력한 BTC 종목 유지
        "default_price": price,  # 입력한 초기자금 유지
        # "default_weight": ",".join(map(str(weight))),
        "error_message": error_message,
    }
    return render(request, "portfolio.html", context)


def corr(request):
    file_path = os.path.join(settings.BASE_DIR, "data", "USD_경제통합.csv")
    df = pd.read_csv(file_path, index_col="Date")
    df_mean = df.groupby("ym").mean()

    mm = MinMaxScaler()
    df_scaled = mm.fit_transform(df_mean)
    df_scaled = pd.DataFrame(df_scaled)
    df_scaled.columns = df_mean.columns
    df_scaled.index = df_mean.index

    df1 = df.loc["2012-11-28":"2016-07-08"].groupby("ym").mean()
    df2 = df.loc["2016-07-09":"2020-05-10"].groupby("ym").mean()
    df3 = df.loc["2020-05-11":"2024-04-19"].groupby("ym").mean()
    df4 = df.loc["2024-04-20":].groupby("ym").mean()

    # 전체기간 상관관계 heatmap
    full_corr_fig = px.imshow(
        df_mean.corr(numeric_only=True, method="spearman"),
        text_auto=".2",
        aspect="auto",
        color_continuous_scale="PuBU",
    )
    full_corr_fig.update_layout(width=700, height=600)

    # 반감기별 상관관계
    df_halving = pd.DataFrame()

    halving = [df1, df2, df3, df4]

    for i, df_corr in enumerate(halving, start=1):
        corr = df_corr.corr(numeric_only=True, method="spearman")["btc"]
        df_halving[f"{i}차 반감기"] = corr

    # 주식시장 그래프
    stock_fig = make_subplots(rows=2, cols=1)
    stock_fig.update_layout(width=1000, height=400)

    # line 그래프프
    for i in df_scaled[["btc", "w5000", "buffet"]]:
        stock_fig.add_trace(
            go.Scatter(x=df_scaled.index, y=df_scaled[i], name=f"{i}"), row=1, col=1
        )
    # bar그래프
    for i in df_halving.loc[["w5000", "buffet"], :]:
        stock_fig.add_trace(
            go.Bar(
                x=df_halving.loc[["w5000", "buffet"], :].index,
                y=df_halving.loc[["w5000", "buffet"], i],
                name=f"{i}",
            ),
            row=2,
            col=1,
        )

    # 경제 성장 그래프(gdp, 경제성장률)
    economy_fig = make_subplots(rows=2, cols=1)
    economy_fig.update_layout(width=1000, height=400)

    # line 그래프
    for i in df_scaled[["btc", "gdp", "경제성장률(USD)"]]:
        economy_fig.add_trace(
            go.Scatter(x=df_scaled.index, y=df_scaled[i], name=f"{i}"), row=1, col=1
        )
    # bar그래프
    for i in df_halving.loc[["gdp", "경제성장률(USD)"], :]:
        economy_fig.add_trace(
            go.Bar(
                x=df_halving.loc[["gdp", "경제성장률(USD)"], :].index,
                y=df_halving.loc[["gdp", "경제성장률(USD)"], i],
                name=f"{i}",
            ),
            row=2,
            col=1,
        )

    # 물가 그래프
    price_fig = make_subplots(rows=2, cols=1)
    price_fig.update_layout(width=1000, height=400)

    # line 그래프프
    for i in df_scaled[
        ["btc", "소비자물가지수(USD)", "생산자물가지수(USD)", "물가상승률(USD)"]
    ]:
        price_fig.add_trace(
            go.Scatter(x=df_scaled.index, y=df_scaled[i], name=f"{i}"), row=1, col=1
        )

    # bar그래프
    for i in df_halving.loc[
        ["소비자물가지수(USD)", "생산자물가지수(USD)", "물가상승률(USD)"], :
    ]:
        price_fig.add_trace(
            go.Bar(
                x=df_halving.loc[
                    ["소비자물가지수(USD)", "생산자물가지수(USD)", "물가상승률(USD)"], :
                ].index,
                y=df_halving.loc[
                    ["소비자물가지수(USD)", "생산자물가지수(USD)", "물가상승률(USD)"], i
                ],
                name=f"{i}",
            ),
            row=2,
            col=1,
        )

    # 통화 정책 그래프
    monetary_fig = make_subplots(rows=2, cols=1)
    monetary_fig.update_layout(width=1000, height=400)

    for i in df_scaled[["btc", "기준금리(USD)", "통화량(USD)", "환율"]]:
        monetary_fig.add_trace(
            go.Scatter(x=df_scaled.index, y=df_scaled[i], name=f"{i}"), row=1, col=1
        )

    # bar그래프
    for i in df_halving.loc[["기준금리(USD)", "통화량(USD)", "환율"], :]:
        monetary_fig.add_trace(
            go.Bar(
                x=df_halving.loc[["기준금리(USD)", "통화량(USD)", "환율"], :].index,
                y=df_halving.loc[["기준금리(USD)", "통화량(USD)", "환율"], i],
                name=f"{i}",
            ),
            row=2,
            col=1,
        )

    # 안전 자산 그래프
    asset_fig = make_subplots(rows=2, cols=1)
    asset_fig.update_layout(width=1000, height=400)
    for i in df_scaled[["btc", "금가격", "채권(USD)"]]:
        asset_fig.add_trace(
            go.Scatter(x=df_scaled.index, y=df_scaled[i], name=f"{i}"), row=1, col=1
        )

    # bar그래프
    for i in df_halving.loc[["금가격", "채권(USD)"], :]:
        asset_fig.add_trace(
            go.Bar(
                x=df_halving.loc[["금가격", "채권(USD)"], :].index,
                y=df_halving.loc[["금가격", "채권(USD)"], i],
                name=f"{i}",
            ),
            row=2,
            col=1,
        )

    # 일반 요청
    full_corr_graph_html = full_corr_fig.to_html(full_html=False)
    stock_graph_html = stock_fig.to_html(full_html=False)
    economy_graph_html = economy_fig.to_html(full_html=False)
    price_graph_html = price_fig.to_html(full_html=False)
    monetary_graph_html = monetary_fig.to_html(full_html=False)
    asset_graph_html = asset_fig.to_html(full_html=False)

    context = {
        "full_corr_graph": full_corr_graph_html,
        "stock_graph": stock_graph_html,
        "economy_graph": economy_graph_html,
        "price_graph": price_graph_html,
        "monetary_graph": monetary_graph_html,
        "asset_graph": asset_graph_html,
    }
    return render(request, "corr.html", context)
