# app_cloud_advisory.py
# Dashboard Azioni Avanzata con Suggerimenti (Compra/Mantieni/Vendi)
# Versione avanzata, robusta per Streamlit Cloud

import time
import logging
from requests.exceptions import RequestException

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup

# -------------------------
# Config e logger
# -------------------------
st.set_page_config(page_title="Dashboard Azioni Avanzata - Advisory", layout="wide")
logger = logging.getLogger("stock_dashboard")
logger.setLevel(logging.INFO)

# -------------------------
# Helper HTTP robusto
# -------------------------
def safe_get(url, headers=None, timeout=8, max_retries=3, backoff_factor=1.8):
    headers = headers or {"User-Agent": "Mozilla/5.0 (compatible; StockDashboard/1.0; +https://example.com)"}
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            else:
                logger.warning(f"safe_get: status {resp.status_code} for {url}")
                return None
        except RequestException as e:
            logger.warning(f"safe_get attempt {attempt} failed for {url}: {e}")
            if attempt == max_retries:
                return None
            time.sleep(delay)
            delay *= backoff_factor
    return None

@st.cache_data(ttl=3600)
def fetch_url_cached(url):
    return safe_get(url)

# -------------------------
# Recupero ticker S&P500
# -------------------------
@st.cache_data(ttl=86400)
def get_sp500_tickers(limit=None):
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = [t.replace('.', '-') for t in table['Symbol'].tolist()]
        if limit:
            return tickers[:limit]
        return tickers
    except Exception as e:
        logger.exception(f"get_sp500_tickers failed: {e}")
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "TSLA"]

# -------------------------
# Indicatori tecnici
# -------------------------
def calcola_indicatore_tecnico(data):
    df = data.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    EMA12 = df['Close'].ewm(span=12, adjust=False).mean()
    EMA26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA12 - EMA26
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# -------------------------
# ARIMAX / ARIMA fallback
# -------------------------
def calcola_trend_arimax(prices, sentiment_series, giorni_previsione=10):
    try:
        exog = np.array(sentiment_series).reshape(-1, 1)
        model = ARIMA(endog=prices, exog=exog, order=(5,1,0))
        model_fit = model.fit()
        exog_forecast = np.repeat(exog[-1, :], giorni_previsione).reshape(-1, 1)
        forecast = model_fit.forecast(steps=giorni_previsione, exog=exog_forecast)
        trend = 1 if forecast.iloc[-1] > prices.iloc[-1] else 0
        return trend, forecast
    except Exception as e:
        logger.warning(f"ARIMAX failed: {e}. Trying ARIMA fallback.")
        try:
            model = ARIMA(prices, order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=giorni_previsione)
            trend = 1 if forecast.iloc[-1] > prices.iloc[-1] else 0
            return trend, forecast
        except Exception as e2:
            logger.warning(f"ARIMA fallback failed: {e2}. Using flat forecast.")
            last = prices.iloc[-1]
            idx = pd.date_range(start=prices.index[-1] + pd.Timedelta(1, unit='D'),
                                periods=giorni_previsione)
            forecast = pd.Series([last]*giorni_previsione, index=idx)
            return 0, forecast

# -------------------------
# Monte Carlo semplice
# -------------------------
def monte_carlo_simulation(last_price, mu, sigma, giorni, n_simulazioni):
    rng = np.random.default_rng(42)
    sim = np.zeros((giorni, n_simulazioni))
    for j in range(n_simulazioni):
        prices = [last_price]
        for i in range(1, giorni):
            shock = rng.normal(mu, sigma)
            prices.append(prices[-1] * np.exp(shock))
        sim[:, j] = prices
    return sim

# -------------------------
# Recupero notizie + sentiment
# -------------------------
def get_news_sentiment(ticker, max_items=3):
    try:
        url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"
        html = fetch_url_cached(url)
        if not html:
            return 0.0, []
        soup = BeautifulSoup(html, "html.parser")
        candidates = soup.find_all(['h3', 'h2', 'a', 'li'])
        news_list = []
        sentiment_totale = 0.0
        count = 0
        for item in candidates:
            if count >= max_items:
                break
            text = item.get_text().strip()
            if not text or len(text) < 12:
                continue
            link = ""
            a = item.find('a')
            if a and a.has_attr('href'):
                link = a['href']
                if link.startswith('/'):
                    link = "https://finance.yahoo.com" + link
            try:
                score = TextBlob(text).sentiment.polarity
            except Exception:
                score = 0.0
            sentiment_totale += score
            news_list.append({"title": text, "link": link, "score": score})
            count += 1
        return sentiment_totale, news_list
    except Exception as e:
        logger.exception(f"get_news_sentiment error for {ticker}: {e}")
        return 0.0, []

# -------------------------
# Valutazione singolo titolo
# -------------------------
def valuta_titolo(ticker, giorni_previsione, n_sim):
    try:
        data = yf.download(ticker, start='2020-01-01', progress=False)
        if data is None or data.empty:
            return None
        df_tecnico = calcola_indicatore_tecnico(data)
        sentiment_score, news_list = get_news_sentiment(ticker)
        sentiment_series = pd.Series([sentiment_score]*len(data), index=data.index)
        trend, forecast = calcola_trend_arimax(data['Close'], sentiment_series, giorni_previsione)
        try:
            returns = np.log(data['Close'] / data['Close'].shift(1)).dropna()
            mu = returns.mean()
            sigma = returns.std()
            mc_sim = monte_carlo_simulation(data['Close'].iloc[-1], mu, sigma, giorni_previsione, n_sim)
            prob_rialzo = (mc_sim[-1, :] > data['Close'].iloc[-1]).mean()
        except Exception:
            mc_sim = np.zeros((giorni_previsione, n_sim))
            prob_rialzo = 0.0
        punteggio = 0.0
        punteggio += trend*2
        rsi = df_tecnico['RSI'].iloc[-1]
        if np.isfinite(rsi):
            if rsi < 30: punteggio += 1
            elif rsi > 70: punteggio -= 1
        macd = df_tecnico['MACD'].iloc[-1]
        signal = df_tecnico['Signal'].iloc[-1]
        if np.isfinite(macd) and np.isfinite(signal):
            punteggio += 1 if macd > signal else -1
        punteggio += float(sentiment_score)
        punteggio += float(prob_rialzo)
        try:
            info = yf.Ticker(ticker).info
        except Exception:
            info = {}
        fundamentals = {
            'P/E': info.get('trailingPE', None) if isinstance(info, dict) else None,
            'EPS': info.get('trailingEps', None) if isinstance(info, dict) else None,
            'Market Cap': info.get('marketCap', None) if isinstance(info, dict) else None,
            'Dividendi': info.get('dividendRate', None) if isinstance(info, dict) else None
        }
        return {"ticker": ticker, "score": punteggio, "df_tecnico": df_tecnico,
                "forecast": forecast, "news": news_list, "mc_sim": mc_sim,
                "fundamentals": fundamentals}
    except Exception as e:
        logger.exception(f"valuta_titolo fatal error for {ticker}: {e}")
        return None

# -------------------------
# Suggerimento Compra/Mantieni/Vendi
# -------------------------
def genera_suggerimento(punteggio, soglia_alta, soglia_bassa):
    if punteggio >= soglia_alta: return "✅ Compra"
    elif punteggio <= soglia_bassa: return "❌ Vendi"
    else: return "⚖️ Mantieni"

# -------------------------
# Ranking globale
# -------------------------
def ranking_globale(tickers, top_n, giorni_previsione, n_sim, delay_between=0.25):
    risultati = []
    dettagli = {}
    total = len(tickers)
    progress = st.progress(0)
    i = 0
    for ticker in tickers:
        i += 1
        progress.progress(int(i/total*100))
        res = valuta_titolo(ticker, giorni_previsione, n_sim)
        if res:
            risultati.append({"Ticker": ticker, "Punteggio": res["score"], **res["fundamentals"]})
            dettagli[ticker] = res
        time.sleep(delay_between)
    progress.empty()
    if not risultati: return pd.DataFrame(), {}
    df_ranking = pd.DataFrame(risultati)
    soglia_alta = df_ranking['Punteggio'].quantile(0.75)
    soglia_bassa = df_ranking['Punteggio'].quantile(0.25)
    df_ranking['Suggerimento'] = df_ranking['Punteggio'].apply(lambda x: genera_suggerimento(x, soglia_alta, soglia_bassa))
    df_ranking = df_ranking.sort_values(by='Punteggio', ascending=False).head(top_n)
    return df_ranking, {t: dettagli[t] for t in df_ranking['Ticker'] if t in dettagli}

# -------------------------
# Interfaccia utente Streamlit
# -------------------------
st.title("📈 Dashboard Azioni Avanzata — Advisory")
st.markdown("Semplice: scegli i parametri e premi **Aggiorna Dashboard**.")

col1, col2, col3 = st.columns([1,1,1])
with col1:
    top_n = st.slider("Top titoli da mostrare", 5, 10, 5)
with col2:
    giorni_previsione = st.slider("Giorni previsione", 5, 30, 10)
with col3:
    simulazioni_mc = st.slider("Simulazioni Monte Carlo", 100, 500, 200)

limit_tickers = st.selectbox("Numero di ticker da analizzare", [30,50,100,250], index=1)
st.caption("Consiglio: 50 è un buon compromesso per Streamlit Cloud gratuito.")

if st.button("Aggiorna Dashboard"):
    try:
        st.info("Avvio analisi... attendere qualche decina di secondi.")
        tickers = get_sp500_tickers(limit=limit_tickers)
        ranking, dettagli = ranking_globale(tickers, top_n, giorni_previsione, simulazioni_mc, delay_between=0.25)
        if ranking.empty:
            st.warning("Nessun risultato disponibile.")
        else:
            st.success("Analisi completata!")
            st.subheader("Top titoli con suggerimento")
            st.dataframe(ranking.reset_index(drop=True))

            st.subheader("Dettagli e grafici")
            for ticker, dati in dettagli.items():
                st.markdown(f"### {ticker}")
                df_tecnico = dati['df_tecnico']
                forecast = dati['forecast']
                fig, ax = plt.subplots(figsize=(8,3))
                ax.plot(df_tecnico['Close'], label='Prezzo storico')
                try:
                    ax.plot(forecast.index, forecast, label='Forecast ARIMAX', color='red')
                except Exception:
                    ax.plot(range(len(forecast)), forecast, label='Forecast', color='red')
                ax.plot(df_tecnico['SMA20'], label='SMA20', color='green')
                ax.plot(df_tecnico['SMA50'], label='SMA50', color='orange')
                ax.legend(loc='upper left', fontsize='small')
                st.pyplot(fig)

                st.markdown("**Notizie recenti:**")
                if dati['news']:
                    for news in dati['news']:
                        title = news.get('title','')[:200]
                        link = news.get('link') or ""
                        score = news.get('score',0.0)
                        if link:
                            st.markdown(f"- [{title}]({link}) → Sentiment: {score:.2f}")
                        else:
                            st.markdown(f"- {title} → Sentiment: {score:.2f}")
                else:
                    st.markdown("- Nessuna notizia disponibile")

                try:
                    prob = (dati['mc_sim'][-1,:] > df_tecnico['Close'].iloc[-1]).mean()
                    st.markdown(f"**Probabilità rialzo (Monte Carlo):** {prob*100:.2f}%")
                except Exception:
                    st.markdown("**Probabilità Monte Carlo non disponibile**")

    except Exception as e:
        logger.exception(f"Dashboard update error: {e}")
        st.error("Si è verificato un errore durante l'aggiornamento. Controlla i log di Streamlit Cloud.")
      
