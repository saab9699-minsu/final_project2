from bs4 import BeautifulSoup
import requests
from urllib.request import urlopen
import pandas as pd
import pyupbit

def news_crawling():
    url = "https://kr.investing.com/news/cryptocurrency-news"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'}

    title_list = []
    company_list = []
    date_list = []
    href_list = []

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # HTTP 오류 발생 시 예외를 발생시킴
        soup = BeautifulSoup(response.text, "html.parser")
        for i in soup.select("div.min-w-0 div > div > ul > li"):
            title_list.append(i.select("a")[0].text.strip())
            company_list.append(i.select("li span")[1].text)
            date_list.append(i.select("li time")[0].text)
            href_list.append(i.select("a")[0]["href"])


    except requests.exceptions.HTTPError as e:
        print(f"HTTPError: {e.response.status_code} - {e.response.reason}")
    except Exception as e:
        print(f"Error: {e}")

    df = pd.DataFrame()
    df["title"] = title_list
    df["company"] = company_list
    df["date"] = date_list
    df["href"] = href_list
    return df

def upbit():
    df = pyupbit.get_ohlcv("KRW-BTC", interval = "mintue5", count = 24 * 12)
    return df