from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen
import pandas as pd
import pyupbit

def news_crawling():
    url = "https://kr.investing.com/news/cryptocurrency-news"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'}

    news_list = []

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        num = 0
        for i in soup.select("div.min-w-0 div > div > ul > li"):
            news_list.append({
                "title" : i.select("a")[0].text.strip(),
                "company" : i.select("li span")[1].text,
                "date" : i.select("li time")[0].text,
                "href" : i.select("a")[0]["href"],
            })
            num += 1

    except:
        pass

    return news_list

def upbit():
    df = pyupbit.get_ohlcv("KRW-BTC", interval = "day", count = 20)
    return df



def upbit2():
    coin_list = ["KRW-BTC", "KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"]
    coin_name = ["비트코인(BTC)", "이더리움(ETH)", "리플(XRP)", "솔라나(SOL)", "도지코인(DOGE)"]
    coin_data = []
    for coin, name in zip(coin_list, coin_name):
        btc_df = pyupbit.get_ohlcv(coin, interval = "day", count = 2)
        coin_data.append(({
            "name" : name,
            "price" : btc_df["close"].iloc[1],
            "change" : (btc_df["close"].pct_change() * 100).iloc[1].round(2),
        }))
    return coin_data