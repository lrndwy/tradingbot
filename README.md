# Advanced Bitcoin Trading Bot

This project implements an advanced Bitcoin trading bot in Python, designed to automate trading decisions based on a combination of technical analysis, market sentiment, and robust risk management. The bot integrates with cryptocurrency exchanges via `ccxt`, stores data in SQLite, and provides real-time notifications.

## Features

*   **Multi-Strategy Signal Generation**: Combines various strategies (Trend Following, Mean Reversion, Momentum, Sentiment) to generate comprehensive trading signals.
*   **Technical Analysis**: Calculates a wide range of indicators including SMA, EMA, RSI, MACD, Bollinger Bands, and Stochastic Oscillator.
*   **Sentiment Analysis**: Integrates with external APIs to fetch Fear & Greed Index and analyze cryptocurrency news sentiment.
*   **Robust Risk Management**: Implements dynamic position sizing, stop-loss, and take-profit calculations based on ATR and risk-reward ratios.
*   **Database Management**: Stores market data, trading signals, and trade history in a SQLite database for backtesting and analysis.
*   **Real-time Notifications**: Sends trade signals, execution alerts, position updates, and error notifications via Telegram and webhooks.
*   **Configurable Exchange Integration**: Supports various cryptocurrency exchanges via `ccxt` (e.g., Binance) with sandbox mode for testing.
*   **Scheduled Operations**: Runs analysis cycles and monitors positions at configurable intervals (e.g., every hour).
*   **Detailed Logging**: Utilizes `colorlog` for enhanced console output and standard file logging for debugging and monitoring.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/tradingbot.git
    cd tradingbot
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Create a `.env` file in the root directory of the project based on the `.env.example` file.

```
# Exchange API Credentials
EXCHANGE_API_KEY=your_exchange_api_key
EXCHANGE_SECRET=your_exchange_secret

# News API (for sentiment analysis)
NEWS_API_KEY=your_news_api_key # Get one from newsapi.org

# Telegram Notification (Optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Database settings
DATABASE_URL=sqlite:///trading_bot.db

# Trading settings
MAX_RISK_PER_TRADE=0.02
MAX_TOTAL_RISK=0.10
MIN_CONFIDENCE=0.70

# Exchange settings
USE_SANDBOX=true
DEFAULT_SYMBOL=BTC/USDT
DEFAULT_TIMEFRAME=1h

```

## Usage

To start the trading bot, run the `main.py` script:

```bash
python main.py
```

The bot will start fetching market data, generating signals, and executing trades based on the configured strategies and risk parameters. It will also send notifications through the enabled channels.

To stop the bot, press `Ctrl+C` in the terminal.

## Project Structure

*   `main.py`: The core script containing the `AdvancedTradingBot` class, strategy implementations, and the main execution loop.
*   `notifications.py`: Handles sending various types of alerts and notifications (Telegram, webhooks).
*   `trading_bot.db`: SQLite database file where market data, signals, and trades are stored.
*   `trading_bot.log`: Log file for detailed bot operations.
*   `.env.example`: Example environment variables file.
*   `requirements.txt`: List of Python dependencies.

## Logging

The bot uses `colorlog` for colored console output and also writes detailed logs to `trading_bot.log`.
