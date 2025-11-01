# src/data_loader.py - ENHANCED WITH MULTIPLE FREE APIs
import os
import json
import requests
import time
import random
from datetime import datetime, timedelta, timezone
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from yahooquery import Ticker as YQ_Ticker
except ImportError:
    YQ_Ticker = None

MODEL_DIR = "models"

# ----------------------------
# ALL FREE DATA PROVIDERS
# Priority Order: Yahoo > Yahoo Query > Polygon.io Free > Alpha Vantage > Twelve Data > Finnhub > CoinGecko
# ----------------------------

SUPPORTED_INTERVALS = [
    "1m", "2m", "5m", "15m", "30m",
    "60m", "90m", "1h", "1d"
]

# Enhanced API configuration
API_PROVIDERS = {
    "yahoo": {"free": True, "rate_limit": None, "priority": 1},
    "yahooquery": {"free": True, "rate_limit": None, "priority": 2},
    "polygon": {"free": True, "rate_limit": 5, "priority": 3},  # 5 calls/min free tier
    "alphavantage": {"free": True, "rate_limit": 5, "priority": 4},  # 5 calls/min
    "twelvedata": {"free": True, "rate_limit": 8, "priority": 5},  # 8 calls/min
    "finnhub": {"free": True, "rate_limit": 60, "priority": 6},  # 60 calls/min
    "coingecko": {"free": True, "rate_limit": 50, "priority": 7},  # 50 calls/min
    "finage": {"free": True, "rate_limit": 10, "priority": 8},  # 10 calls/min free
    "marketstack": {"free": True, "rate_limit": 100, "priority": 9},  # 100/month free
}

# Ticker normalization mapping
TICKER_MAPPING = {
    # Stocks / ETFs
    "AAPL": {"yahoo": "AAPL", "polygon": "AAPL", "alphavantage": "AAPL"},
    "TSLA": {"yahoo": "TSLA", "polygon": "TSLA", "alphavantage": "TSLA"},
    "SPY": {"yahoo": "SPY", "polygon": "SPY", "alphavantage": "SPY"},
    "QQQ": {"yahoo": "QQQ", "polygon": "QQQ", "alphavantage": "QQQ"},
    "MSFT": {"yahoo": "MSFT", "polygon": "MSFT", "alphavantage": "MSFT"},
    "GOOGL": {"yahoo": "GOOGL", "polygon": "GOOGL", "alphavantage": "GOOGL"},
    "AMZN": {"yahoo": "AMZN", "polygon": "AMZN", "alphavantage": "AMZN"},

    # Indices
    "^GSPC": {"yahoo": "^GSPC"},
    "^IXIC": {"yahoo": "^IXIC"},
    "^DJI": {"yahoo": "^DJI"},
    "^VIX": {"yahoo": "^VIX"},

    # Commodities
    "XAUUSD": {"yahoo": "GC=F", "twelvedata": "XAU/USD", "finnhub": "OANDA:XAU_USD"},
    "XAGUSD": {"yahoo": "SI=F", "twelvedata": "XAG/USD", "finnhub": "OANDA:XAG_USD"},
    "CL=F": {"yahoo": "CL=F"},
    "GC=F": {"yahoo": "GC=F"},
    "SI=F": {"yahoo": "SI=F"},
    "NG=F": {"yahoo": "NG=F"},

    # Forex
    "EURUSD": {"yahoo": "EURUSD=X", "twelvedata": "EUR/USD", "finnhub": "OANDA:EUR_USD", "polygon": "C:EURUSD"},
    "GBPUSD": {"yahoo": "GBPUSD=X", "twelvedata": "GBP/USD", "finnhub": "OANDA:GBP_USD", "polygon": "C:GBPUSD"},
    "USDJPY": {"yahoo": "USDJPY=X", "twelvedata": "USD/JPY", "finnhub": "OANDA:USD_JPY", "polygon": "C:USDJPY"},
    "AUDUSD": {"yahoo": "AUDUSD=X", "twelvedata": "AUD/USD", "finnhub": "OANDA:AUD_USD", "polygon": "C:AUDUSD"},
    "NZDUSD": {"yahoo": "NZDUSD=X", "twelvedata": "NZD/USD", "finnhub": "OANDA:NZD_USD", "polygon": "C:NZDUSD"},
    "USDCAD": {"yahoo": "USDCAD=X", "twelvedata": "USD/CAD", "finnhub": "OANDA:USD_CAD", "polygon": "C:USDCAD"},
    "USDCHF": {"yahoo": "USDCHF=X", "twelvedata": "USD/CHF", "finnhub": "OANDA:USD_CHF", "polygon": "C:USDCHF"},
    
    # Crypto
    "BTCUSD": {"yahoo": "BTC-USD", "twelvedata": "BTC/USD", "finnhub": "BINANCE:BTCUSDT", "coingecko": "bitcoin", "polygon": "X:BTCUSD"},
    "ETHUSD": {"yahoo": "ETH-USD", "twelvedata": "ETH/USD", "finnhub": "BINANCE:ETHUSDT", "coingecko": "ethereum", "polygon": "X:ETHUSD"},
    "BTCUSDT": {"yahoo": "BTC-USD", "twelvedata": "BTC/USD", "finnhub": "BINANCE:BTCUSDT", "coingecko": "bitcoin"},
    "ETHUSDT": {"yahoo": "ETH-USD", "twelvedata": "ETH/USD", "finnhub": "BINANCE:ETHUSDT", "coingecko": "ethereum"},
    "SOLUSDT": {"yahoo": "SOL-USD", "coingecko": "solana"},
    "XRPUSDT": {"yahoo": "XRP-USD", "coingecko": "ripple"},
}

def normalize_ticker(ticker: str, provider: Optional[str] = "yahoo") -> str:
    """
    Enhanced ticker normalization supporting ALL broker formats
    Supports: MT4, MT5, cTrader, TradingView, Interactive Brokers, etc.
    """
    if not isinstance(ticker, str) or ticker.strip() == "":
        return ticker

    ticker = ticker.strip()
    provider = provider or "yahoo"

    # Remove common broker prefixes
    prefixes = ["FXCM:", "OANDA:", "BINANCE:", "FX:", "CRYPTO:", "STOCK:"]
    for prefix in prefixes:
        if ticker.startswith(prefix):
            ticker = ticker.replace(prefix, "")
    
    # Handle uppercase broker tickers
    if ticker.isupper() and len(ticker) in (6, 7, 8):
        # BTCUSDT format
        if ticker.endswith("USDT") and len(ticker) > 4:
            base = ticker[:-4]
            if provider == "yahoo":
                return f"{base}-USD"
            if provider == "twelvedata":
                return f"{base}/USD"
            if provider == "finnhub":
                return f"BINANCE:{base}USDT"
            if provider == "coingecko":
                return base.lower()
            if provider == "polygon":
                return f"X:{base}USD"
            return ticker
        
        # Forex pairs: EURUSD, GBPJPY, etc.
        if len(ticker) == 6:
            base, quote = ticker[:3], ticker[3:]
            currency_codes = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]
            if base in currency_codes and quote in currency_codes:
                if provider == "yahoo":
                    return f"{base}{quote}=X"
                if provider == "twelvedata":
                    return f"{base}/{quote}"
                if provider == "finnhub":
                    return f"OANDA:{base}_{quote}"
                if provider == "polygon":
                    return f"C:{base}{quote}"
                return ticker
        
        # Metals: XAUUSD, XAGUSD
        if ticker.startswith("XAU") and ticker.endswith("USD"):
            if provider == "yahoo":
                return "GC=F"
            if provider == "twelvedata":
                return "XAU/USD"
            if provider == "finnhub":
                return "OANDA:XAU_USD"
            return ticker
        
        if ticker.startswith("XAG") and ticker.endswith("USD"):
            if provider == "yahoo":
                return "SI=F"
            if provider == "twelvedata":
                return "XAG/USD"
            if provider == "finnhub":
                return "OANDA:XAG_USD"
            return ticker
        
        # Crypto without USDT: BTCUSD, ETHUSD
        crypto_bases = ["BTC", "ETH", "SOL", "ADA", "XRP", "LTC", "DOGE", "LINK", "DOT", "UNI"]
        if ticker.endswith("USD") and len(ticker) > 3:
            base = ticker[:-3]
            if base in crypto_bases:
                if provider == "yahoo":
                    return f"{base}-USD"
                if provider == "twelvedata":
                    return f"{base}/USD"
                if provider == "finnhub":
                    return f"BINANCE:{base}USDT"
                if provider == "coingecko":
                    return base.lower()
                if provider == "polygon":
                    return f"X:{base}USD"
                return ticker

    # TradingView format: "BINANCE:BTCUSDT"
    if ":" in ticker:
        prefix, symbol = ticker.split(":", 1)
        symbol = symbol.strip()
        
        if symbol.endswith("USDT"):
            base = symbol.replace("USDT", "")
            if provider == "yahoo":
                return f"{base}-USD"
            if provider == "coingecko":
                return base.lower()
        
        if len(symbol) == 6:  # Forex
            base, quote = symbol[:3], symbol[3:]
            if provider == "yahoo":
                return f"{base}{quote}=X"
            if provider == "twelvedata":
                return f"{base}/{quote}"
        
        return symbol

    # Check static mapping
    ticker_upper = ticker.upper()
    if ticker_upper in TICKER_MAPPING:
        mapped = TICKER_MAPPING[ticker_upper]
        if provider in mapped:
            return mapped[provider]
    
    # Cross pairs: ETH/BTC
    if "/" in ticker:
        base, quote = ticker.split("/")
        if provider == "twelvedata":
            return f"{base}/{quote}"
        if provider == "finnhub":
            return f"BINANCE:{base}{quote}"
        if provider == "coingecko":
            return base.lower()
        return ticker

    # Forex with =X suffix
    if ticker.endswith("=X"):
        base_pair = ticker.replace("=X", "")
        if len(base_pair) >= 6:
            base, quote = base_pair[:3], base_pair[3:]
            if provider == "twelvedata":
                return f"{base}/{quote}"
            if provider == "finnhub":
                return f"OANDA:{base}_{quote}"
            if provider == "polygon":
                return f"C:{base}{quote}"
        return ticker

    # Crypto with -USD suffix
    if "-USD" in ticker:
        symbol = ticker.split("-")[0]
        if provider == "twelvedata":
            return f"{symbol}/USD"
        if provider == "finnhub":
            return f"BINANCE:{symbol}USDT"
        if provider == "coingecko":
            return symbol.lower()
        if provider == "polygon":
            return f"X:{symbol}USD"
        return ticker

    return ticker

# Rate limiting
def rate_limited_request(func, *args, max_retries=3, base_delay=1.0, **kwargs):
    """Execute request with exponential backoff"""
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                print(f"[INFO] Waiting {delay:.1f}s before retry {attempt + 1}...")
                time.sleep(delay)
            else:
                time.sleep(random.uniform(0.2, 0.8))
            
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            print(f"[WARN] Attempt {attempt + 1} failed: {str(e)[:100]}...")
    return None

# Data validation
def validate_data(df: pd.DataFrame, ticker: str) -> bool:
    if df is None or df.empty:
        raise ValueError(f"No data available for {ticker}")
    
    if len(df) < 50:
        raise ValueError(f"Insufficient data for {ticker}: only {len(df)} rows (need at least 50)")
    
    if 'y' not in df.columns:
        raise ValueError(f"Missing price column 'y' for {ticker}")
    
    if df['y'].isna().sum() > len(df) * 0.5:
        raise ValueError(f"Too many missing values in {ticker} data: {df['y'].isna().sum()}/{len(df)}")
    
    return True

def normalize_df_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize dataframe columns"""
    if df.empty:
        return df
    
    date_cols = ["Date", "Datetime", "date", "datetime", "timestamp", "ds", "time"]
    for col in date_cols:
        if col in df.columns:
            df = df.rename(columns={col: "ds"})
            break
    else:
        if df.index.name:
            df = df.reset_index().rename(columns={df.index.name: "ds"})
        else:
            df["ds"] = pd.to_datetime(df.index, utc=True)

    price_cols = ["Close", "close", "CLOSE", "price", "Price", "last"]
    for col in price_cols:
        if col in df.columns:
            df = df.rename(columns={col: "y"})
            break

    df["ds"] = pd.to_datetime(df["ds"], utc=True)
    df = df.sort_values("ds").reset_index(drop=True)

    standard_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in standard_cols:
        if col not in df.columns:
            df[col] = np.nan

    if "y" in df.columns:
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

    return df

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators with ATR"""
    if df.empty:
        return df
    
    df = df.copy()
    price_col = "Close" if "Close" in df.columns else "y"
    df["y"] = pd.to_numeric(df[price_col], errors="coerce").astype(float)

    try:
        df["return"] = df["y"].pct_change()
        df["ma_5"] = df["y"].rolling(window=5, min_periods=1).mean()
        df["ma_20"] = df["y"].rolling(window=20, min_periods=1).mean()
        df["vol_10"] = df["y"].rolling(window=10, min_periods=1).std()

        # RSI
        delta = df["y"].diff()
        up = delta.clip(lower=0).rolling(window=14, min_periods=1).mean()
        down = (-delta.clip(upper=0)).rolling(window=14, min_periods=1).mean()
        rs = up / (down.replace(0, np.nan))
        df["rsi"] = 100 - (100 / (1 + rs))

        # ATR calculation
        try:
            if all(col in df.columns for col in ["High", "Low", "Close"]):
                high = pd.to_numeric(df["High"], errors="coerce")
                low = pd.to_numeric(df["Low"], errors="coerce")
                close = pd.to_numeric(df["Close"], errors="coerce")
                prev_close = close.shift(1)
                
                tr1 = (high - low).abs()
                tr2 = (high - prev_close).abs()
                tr3 = (low - prev_close).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                
                atr = tr.rolling(window=14, min_periods=max(1, 7)).mean()
                df["atr_14"] = atr.fillna(method="bfill").fillna(method="ffill")
                df["atr_pct_14"] = (atr / close * 100.0).fillna(method="bfill").fillna(method="ffill").fillna(1.0)
            else:
                df["atr_14"] = df["y"] * 0.01
                df["atr_pct_14"] = 1.0
        except Exception as e:
            print(f"[WARN] ATR calculation failed: {e}")
            df["atr_14"] = df["y"] * 0.01
            df["atr_pct_14"] = 1.0

        df = df.fillna(method="bfill").fillna(method="ffill")
        
        for col in ["return", "ma_5", "ma_20", "vol_10", "rsi", "atr_14", "atr_pct_14"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        
    except Exception as e:
        print(f"[WARN] Technical indicators failed: {e}")
        for col in ["return", "ma_5", "ma_20", "vol_10", "rsi", "atr_14", "atr_pct_14"]:
            if col not in df.columns:
                df[col] = 0.0

    return df

# ----------------------------
# ENHANCED DATA PROVIDERS
# ----------------------------

def try_yahoo_finance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Yahoo Finance (Primary free source)"""
    def fetch():
        yf_ticker = normalize_ticker(ticker, "yahoo")
        return yf.Ticker(yf_ticker).history(period=period, interval=interval)
    
    df = rate_limited_request(fetch)
    if df is not None and not df.empty:
        return normalize_df_columns(df.reset_index())
    return pd.DataFrame()

def try_yahooquery(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Yahoo Query (Secondary free source)"""
    if not YQ_Ticker:
        return pd.DataFrame()
    
    def fetch():
        yq_ticker = normalize_ticker(ticker, "yahoo")
        yq = YQ_Ticker(yq_ticker)
        return yq.history(period=period, interval=interval)
    
    df = rate_limited_request(fetch)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df = df.reset_index()
        if "date" in df.columns:
            df = df.rename(columns={"date": "ds", "close": "y"})
        df["ds"] = pd.to_datetime(df["ds"], utc=True)
        return df.sort_values("ds").reset_index(drop=True)
    return pd.DataFrame()

def try_polygon(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Polygon.io (Free tier: 5 calls/min)"""
    api_key = os.getenv("POLYGON_API_KEY", "").strip()
    if not api_key:
        return pd.DataFrame()
    
    def fetch():
        poly_ticker = normalize_ticker(ticker, "polygon")
        
        # Map interval
        if interval.endswith("m"):
            multiplier = int(interval[:-1])
            timespan = "minute"
        elif interval.endswith("h"):
            multiplier = int(interval[:-1])
            timespan = "hour"
        elif interval.endswith("d"):
            multiplier = int(interval[:-1])
            timespan = "day"
        else:
            multiplier = 1
            timespan = "day"
        
        # Calculate date range
        end_date = datetime.now()
        if period.endswith("y"):
            years = int(period[:-1])
            start_date = end_date - timedelta(days=365 * years)
        elif period.endswith("mo"):
            months = int(period[:-2])
            start_date = end_date - timedelta(days=30 * months)
        else:
            start_date = end_date - timedelta(days=365)
        
        url = f"https://api.polygon.io/v2/aggs/ticker/{poly_ticker}/range/{multiplier}/{timespan}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        
        res = requests.get(url, params={"apiKey": api_key, "limit": 50000}, timeout=30)
        res.raise_for_status()
        return res.json()
    
    data = rate_limited_request(fetch)
    if data and "results" in data:
        df = pd.DataFrame(data["results"])
        df["ds"] = pd.to_datetime(df["t"], unit="ms", utc=True)
        df = df.rename(columns={"c": "y", "o": "Open", "h": "High", "l": "Low", "v": "Volume"})
        return df[["ds", "y", "Open", "High", "Low", "Volume"]].sort_values("ds").reset_index(drop=True)
    return pd.DataFrame()

def try_finage(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Finage.co.uk (Free tier: 10 calls/min)"""
    api_key = os.getenv("FINAGE_API_KEY", "").strip()
    if not api_key:
        return pd.DataFrame()
    
    def fetch():
        # Finage uses different endpoints for different asset classes
        if ticker.endswith("=X") or len(ticker) == 6:  # Forex
            endpoint = f"https://api.finage.co.uk/last/forex/{ticker.replace('=X', '')}"
        elif "-USD" in ticker:  # Crypto
            crypto_symbol = ticker.replace("-USD", "")
            endpoint = f"https://api.finage.co.uk/last/crypto/{crypto_symbol}USD"
        else:  # Stocks
            endpoint = f"https://api.finage.co.uk/last/stock/{ticker}"
        
        res = requests.get(endpoint, params={"apikey": api_key}, timeout=30)
        res.raise_for_status()
        return res.json()
    
    data = rate_limited_request(fetch)
    if data:
        # Finage free tier has limitations, this is a simplified implementation
        df = pd.DataFrame([{
            "ds": datetime.now(timezone.utc),
            "y": data.get("price", 0),
            "Open": data.get("price", 0),
            "High": data.get("price", 0),
            "Low": data.get("price", 0),
            "Volume": 0
        }])
        return df
    return pd.DataFrame()

def try_marketstack(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Marketstack (Free tier: 100 calls/month)"""
    api_key = os.getenv("MARKETSTACK_API_KEY", "").strip()
    if not api_key:
        return pd.DataFrame()
    
    def fetch():
        url = f"https://api.marketstack.com/v1/eod"
        params = {
            "access_key": api_key,
            "symbols": ticker,
            "limit": 1000
        }
        res = requests.get(url, params=params, timeout=30)
        res.raise_for_status()
        return res.json()
    
    data = rate_limited_request(fetch)
    if data and "data" in data:
        df = pd.DataFrame(data["data"])
        df["ds"] = pd.to_datetime(df["date"], utc=True)
        df = df.rename(columns={"close": "y", "open": "Open", "high": "High", "low": "Low", "volume": "Volume"})
        return df[["ds", "y", "Open", "High", "Low", "Volume"]].sort_values("ds").reset_index(drop=True)
    return pd.DataFrame()

def try_twelvedata(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Twelve Data (Free tier: 8 calls/min)"""
    td_key = os.getenv("TWELVEDATA_API_KEY", "").strip()
    if not td_key:
        return pd.DataFrame()
    
    def fetch():
        td_symbol = normalize_ticker(ticker, "twelvedata")
        td_interval = interval.replace("m", "min").replace("h", "h").replace("d", "day")
        
        url = "https://api.twelvedata.com/time_series"
        r = requests.get(url, params={
            "symbol": td_symbol,
            "interval": td_interval,
            "apikey": td_key,
            "outputsize": 5000
        }, timeout=30)
        r.raise_for_status()
        return r.json()
    
    data = rate_limited_request(fetch)
    if data and "values" in data:
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={
            "datetime": "ds", "close": "y",
            "open": "Open", "high": "High",
            "low": "Low", "volume": "Volume"
        })
        df["ds"] = pd.to_datetime(df["ds"], utc=True)
        return df.sort_values("ds").reset_index(drop=True)
    return pd.DataFrame()

def try_finnhub(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Finnhub (Free tier: 60 calls/min)"""
    fh_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if not fh_key:
        return pd.DataFrame()
    
    def fetch():
        fh_symbol = normalize_ticker(ticker, "finnhub")
        
        # Map interval to resolution
        if interval.endswith("m"):
            resolution = interval[:-1]
        elif interval.endswith("h"):
            resolution = str(int(interval[:-1]) * 60)
        elif interval.endswith("d"):
            resolution = "D"
        else:
            resolution = "D"
        
        # Calculate timestamps
        end = int(datetime.now(timezone.utc).timestamp())
        if period.endswith("y"):
            days = int(period[:-1]) * 365
        elif period.endswith("mo"):
            days = int(period[:-2]) * 30
        elif period.endswith("d"):
            days = int(period[:-1])
        else:
            days = 365
        start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())

        # Determine endpoint
        if fh_symbol.startswith("BINANCE:"):
            endpoint = "https://finnhub.io/api/v1/crypto/candle"
        elif fh_symbol.startswith("OANDA:"):
            endpoint = "https://finnhub.io/api/v1/forex/candle"
        else:
            endpoint = "https://finnhub.io/api/v1/stock/candle"

        params = {"symbol": fh_symbol, "resolution": resolution,
                  "from": start, "to": end, "token": fh_key}
        res = requests.get(endpoint, params=params, timeout=30)
        res.raise_for_status()
        return res.json()
    
    data = rate_limited_request(fetch)
    if data and "t" in data and data.get("s") == "ok":
        df = pd.DataFrame({
            "ds": pd.to_datetime(data["t"], unit="s", utc=True),
            "y": data["c"], "Open": data["o"],
            "High": data["h"], "Low": data["l"], "Volume": data["v"]
        })
        return df.sort_values("ds").reset_index(drop=True)
    return pd.DataFrame()

def try_alphavantage(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Alpha Vantage (Free tier: 5 calls/min)"""
    av_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not av_key:
        return pd.DataFrame()
    
    def fetch():
        av_symbol = normalize_ticker(ticker, "alphavantage")
        
        # Map interval to function
        if interval.endswith("m") or interval.endswith("h"):
            function = "TIME_SERIES_INTRADAY"
            av_interval = interval.replace("m", "min").replace("h", "h")
        else:
            function = "TIME_SERIES_DAILY_ADJUSTED"
            av_interval = "daily"

        params = {"symbol": av_symbol, "apikey": av_key, "outputsize": "full"}
        if function == "TIME_SERIES_INTRADAY":
            params["function"] = "TIME_SERIES_INTRADAY"
            params["interval"] = av_interval
        else:
            params["function"] = "TIME_SERIES_DAILY_ADJUSTED"

        res = requests.get("https://www.alphavantage.co/query", params=params, timeout=30)
        res.raise_for_status()
        return res.json()
    
    data = rate_limited_request(fetch)
    if data:
        # Find the time series key
        time_series_key = None
        for key in data.keys():
            if "Time Series" in key:
                time_series_key = key
                break
        
        if time_series_key:
            series = data[time_series_key]
            df = pd.DataFrame.from_dict(series, orient="index")
            
            # Normalize column names
            df = df.rename(columns={
                "5. adjusted close": "y",
                "4. close": "Close",
                "1. open": "Open",
                "2. high": "High", 
                "3. low": "Low",
                "6. volume": "Volume",
                "5. volume": "Volume"
            })
            
            df["ds"] = pd.to_datetime(df.index, utc=True)
            df = df.sort_values("ds").reset_index(drop=True)
            
            if "y" not in df.columns and "Close" in df.columns:
                df["y"] = pd.to_numeric(df["Close"], errors="coerce")
            
            return df
    return pd.DataFrame()

def try_coingecko(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """CoinGecko (Free tier: 50 calls/min)"""
    def fetch():
        cg_symbol = normalize_ticker(ticker, "coingecko")
        
        # Calculate days
        if period.endswith("y"):
            days = int(period[:-1]) * 365
        elif period.endswith("mo"):
            days = int(period[:-2]) * 30
        elif period.endswith("d"):
            days = int(period[:-1])
        else:
            days = 365
        
        days = min(days, 365)  # CoinGecko free tier limit
        
        url = f"https://api.coingecko.com/api/v3/coins/{cg_symbol}/market_chart"
        res = requests.get(url, params={"vs_currency": "usd", "days": days, "interval": "daily"}, timeout=30)
        res.raise_for_status()
        return res.json()
    
    data = rate_limited_request(fetch)
    if data and "prices" in data:
        df = pd.DataFrame(data["prices"], columns=["ds", "y"])
        df["ds"] = pd.to_datetime(df["ds"], unit="ms", utc=True)
        df = df.sort_values("ds").reset_index(drop=True)
        return df
    return pd.DataFrame()

# ----------------------------
# MAIN DOWNLOAD FUNCTION WITH ALL FALLBACKS
# ----------------------------
def download_ticker(ticker: str, period: str = "5y", interval: str = "1d", incremental: bool = True) -> pd.DataFrame:
    """
    Enhanced download with ALL free API providers as fallbacks
    Priority order: Yahoo > Yahoo Query > Polygon > Alpha Vantage > Twelve Data > Finnhub > CoinGecko > Finage > Marketstack
    """
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"Interval '{interval}' not supported. Choose from {SUPPORTED_INTERVALS}")

    print(f"\n[INFO] Fetching {ticker} (period={period}, interval={interval})")
    ticker_dir = os.path.join(MODEL_DIR, ticker)
    os.makedirs(ticker_dir, exist_ok=True)
    csv_path = os.path.join(ticker_dir, f"full_data_{interval}.csv")

    # Provider cascade with ALL free APIs
    providers = [
        ("yahoo", try_yahoo_finance),
        ("yahooquery", try_yahooquery),
        ("polygon", try_polygon),
        ("alphavantage", try_alphavantage),
        ("twelvedata", try_twelvedata),
        ("finnhub", try_finnhub),
        ("coingecko", try_coingecko),
        ("finage", try_finage),
        ("marketstack", try_marketstack),
    ]
    
    last_error = None
    for provider_name, provider_func in providers:
        try:
            print(f"[INFO] Trying {provider_name}...")
            df = provider_func(ticker, period, interval)
            
            if df is not None and not df.empty:
                validate_data(df, ticker)
                df.to_csv(csv_path, index=False)
                print(f"[SUCCESS] {provider_name} returned {len(df)} rows")
                return df
            else:
                print(f"[WARN] {provider_name} returned empty data")
        except Exception as e:
            print(f"[ERROR] {provider_name} failed: {str(e)[:150]}")
            last_error = e
            continue

    # If all providers failed
    error_msg = f"[FAIL] No data available for {ticker} across all APIs (Yahoo, Polygon, Alpha Vantage, Twelve Data, Finnhub, CoinGecko, Finage, Marketstack)."
    if last_error:
        error_msg += f" Last error: {str(last_error)[:200]}"
    raise ValueError(error_msg)

def prepare_features_for_model(df: pd.DataFrame, feature_cols=None) -> Tuple[np.ndarray, pd.DataFrame]:
    """Prepare features for model training"""
    if feature_cols is None:
        feature_cols = ["y", "ma_5", "ma_20", "vol_10", "rsi"]
    
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    
    for col in feature_cols:
        if df[col].isna().all():
            raise ValueError(f"All values are NaN in column:{col}")
    
    values = df[feature_cols].values.astype(float)
    
    if np.isnan(values).any():
        print(f"[WARN] Found NaN values in features, filling with forward fill")
        values = pd.DataFrame(values).ffill().bfill().values
    
    return values, df

