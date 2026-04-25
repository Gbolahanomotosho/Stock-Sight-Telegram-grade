# 📈 Stock-Sight-Telegram-grade

**Advanced Stock Market Telegram Bot** – Real-time prices, technical analysis, and stock grading/rating system delivered directly to your chat.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Telegram](https://img.shields.io/badge/Telegram-Bot_API-blue)
![Yahoo Finance](https://img.shields.io/badge/Yahoo-Finance_API-purple)
![Render](https://img.shields.io/badge/Render-Deployment-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| ⭐ **Stock Grading System** | Rates stocks from A (Strong Buy) to F (Strong Sell) based on multiple indicators |
| 📊 **Technical Analysis** | RSI, Moving Averages, MACD, and volatility calculations |
| 🔔 **Price Alerts** | Get notified when stocks hit your target price |
| 📈 **Real-time Prices** | Live stock data from Yahoo Finance |
| ⭐ **Custom Watchlist** | Track your favorite stocks |
| ☁️ **Cloud Ready** | Deployed on Render (see `render.yaml` & `procfile`) |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Language** | Python 3.11 |
| **Bot Framework** | python-telegram-bot |
| **Data Source** | Yahoo Finance (yfinance) |
| **Deployment** | Render (with `render.yaml` and `procfile`) |
| **Storage** | JSON files (`subscriptions.json`) |

---

## 📂 Project Structure

```

Stock-Sight-Telegram-grade/
├── app.py                # Main bot application
├── main.py               # Entry point
├── web8.py               # Webhook handler for deployment
├── src/                  # Core modules
├── requirements.txt      # Dependencies
├── runtime.txt           # Python version for deployment
├── procfile              # Gunicorn entry for Render
├── render.yaml           # Render deployment config
├── subscriptions.json    # User watchlist storage
└── Required API Ke-WPS Office.txt  # API key template (do not commit real keys)

```

---

## 🔧 Installation & Local Testing

```bash
# Clone the repository
git clone https://github.com/Gbolahanomotosho/Stock-Sight-Telegram-grade.git
cd Stock-Sight-Telegram-grade

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Telegram Bot Token (get from @BotFather)
export TELEGRAM_BOT_TOKEN="your_token_here"

# Run the bot
python app.py
```

---

🤖 Example Bot Commands

Command What it does
/start Welcome message and instructions
/grade AAPL Get stock grade (A-F) for Apple with explanation
/price TSLA Get current Tesla stock price
/watch MSFT Add Microsoft to your watchlist
/alert AAPL 180 Alert me when Apple hits $180
/watchlist Show all your tracked stocks
/analyze GOOGL Get detailed technical analysis (RSI, MACD, etc.)
/remove AAPL Remove from watchlist

---

🧠 What I Built (My Contribution)

· Complete Telegram bot logic – Command handling, user sessions, error recovery
· Stock grading algorithm – Custom rating system from A to F based on:
  · Price vs. 50-day and 200-day moving averages
  · RSI (Relative Strength Index) – oversold/overbought signals
  · MACD trend analysis
  · Volatility assessment
· Yahoo Finance integration – Real-time and historical data
· Alert system – Background price monitoring with instant notifications
· Webhook deployment – Configured for Render cloud platform
· Persistent storage – JSON-based user preferences

---

📈 Stock Grading System Explained

Grade Meaning Criteria
A Strong Buy Price above both MAs, RSI between 40-70, positive MACD
B Buy Price above 50-day MA, RSI 30-70, neutral MACD
C Hold Price trading near MAs, RSI normal range
D Sell Price below 50-day MA, RSI above 70 or below 30
F Strong Sell Price below both MAs, RSI extreme, negative MACD

---

🚧 Current Status & Planned Improvements

Component Status
Real-time price fetching ✅ Complete
Stock grading system ✅ Complete
Technical analysis ✅ Complete
Custom alerts ✅ Complete
Watchlist management ✅ Complete
Webhook deployment ✅ Complete
PostgreSQL migration (replace JSON) 🔄 Planned
More indicators (Bollinger Bands, Fibonacci) 🔄 Planned
Multi-language support (German) 🔄 Planned

---

📈 Why This Matters for German Employers

This project demonstrates:

· ✅ API integration (Telegram + Yahoo Finance)
· ✅ Algorithm development (custom stock grading logic)
· ✅ Data analysis (technical indicators)
· ✅ Asynchronous programming (handling multiple users)
· ✅ Production deployment (Render cloud platform)
· ✅ Real-world utility (investing tools for retail traders)

---

📫 Contact & Visa Status

Omotosho Gbolahan Hammed

· GitHub: Gbolahanomotosho
· Email: hammedg621@gmail.com
· 🛂 German IT Specialist Visa Eligible – 7+ years IT experience. No degree recognition required.

---

📜 License

MIT License – free for personal and commercial use with attribution.
