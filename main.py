import os
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import json
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import requests
from dataclasses import dataclass
import schedule
import os
from dotenv import load_dotenv
import notifications
import colorlog # Import colorlog

# Load environment variables
load_dotenv()

# Configure logging
log_colors = {
    'DEBUG': 'cyan',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}

formatter = colorlog.ColoredFormatter(
    '%(log_color)s[%(asctime)s.%(msecs)03d] [%(levelname)-8s] [%(process)d] [%(module)s:%(funcName)s]%(reset)s \n%(message)s',
    log_colors=log_colors
)

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Create a StreamHandler for console output with color
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Create a FileHandler for log file without color
file_handler = logging.FileHandler('trading_bot.log')
file_handler.setFormatter(logging.For_make_requestmatter('[%(asctime)s] [%(levelname)-8s] [%(process)d] %(message)s'))
logger.addHandler(file_handler)

# Suppress verbose output from ccxt's internal logging
logging.getLogger('ccxt').setLevel(logging.INFO)

# Register datetime adapter/converter for SQLite (Python 3.12+)
import datetime as _dt_mod
sqlite3.register_adapter(_dt_mod.datetime, lambda val: val.isoformat(" "))
sqlite3.register_converter("timestamp", lambda val: _dt_mod.datetime.fromisoformat(val.decode("utf-8")))

@dataclass
class TradingSignal:
    timestamp: datetime
    signal: str  # BUY, SELL, HOLD
    price: float
    confidence: float
    indicators: Dict
    reasoning: str

@dataclass
class Position:
    symbol: str
    side: str  # long/short
    amount: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    current_pnl: float = 0.0

class TechnicalAnalyzer:
    """Advanced technical analysis with multiple indicators"""

    def __init__(self):
        self.indicators = {}

    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=period).mean()

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=period).mean()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
        """MACD Indicator"""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict:
        """Bollinger Bands"""
        sma = self.calculate_sma(prices, period)
        std = prices.rolling(window=period).std()

        return {
            'upper': sma + (std * std_dev),
            'middle': sma,
            'lower': sma - (std * std_dev)
        }

    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14) -> Dict:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()

        return {
            'k_percent': k_percent,
            'd_percent': d_percent
        }

    def analyze_all_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        close = df['close']
        high = df['high']
        low = df['low']

        indicators = {
            'sma_20': self.calculate_sma(close, 20),
            'sma_50': self.calculate_sma(close, 50),
            'ema_12': self.calculate_ema(close, 12),
            'ema_26': self.calculate_ema(close, 26),
            'rsi': self.calculate_rsi(close),
            'macd': self.calculate_macd(close),
            'bollinger': self.calculate_bollinger_bands(close),
            'stochastic': self.calculate_stochastic(high, low, close)
        }

        return indicators

class SentimentAnalyzer:
    """News and social sentiment analysis"""

    def __init__(self):
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.fear_greed_url = "https://api.alternative.me/fng/"

    def get_fear_greed_index(self) -> Optional[Dict]:
        """Get Fear & Greed Index"""
        try:
            response = requests.get(self.fear_greed_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                return {
                    'value': int(data['data'][0]['value']),
                    'classification': data['data'][0]['value_classification']
                }
        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
        return None

    def get_crypto_news_sentiment(self) -> float:
        """Get news sentiment score (-1 to 1)"""
        try:
            if not self.news_api_key:
                return 0.0

            url = "https://newsapi.org/v2/everything"
            params = {
                'q': 'bitcoin OR cryptocurrency',
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': (datetime.now() - timedelta(days=1)).isoformat(),
                'apiKey': self.news_api_key
            }

            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                articles = response.json()['articles'][:10]  # Top 10 recent articles

                # Simple sentiment analysis based on keywords
                positive_words = ['bullish', 'rally', 'surge', 'moon', 'breakthrough', 'adoption', 'institutional']
                negative_words = ['bearish', 'crash', 'dump', 'regulation', 'ban', 'concern', 'volatile']

                sentiment_score = 0
                for article in articles:
                    title = article['title'].lower()
                    description = (article['description'] or '').lower()
                    text = f"{title} {description}"

                    positive_count = sum(1 for word in positive_words if word in text)
                    negative_count = sum(1 for word in negative_words if word in text)

                    sentiment_score += (positive_count - negative_count)

                # Normalize to -1 to 1 range
                max_possible = len(articles) * max(len(positive_words), len(negative_words))
                if max_possible > 0:
                    return max(-1, min(1, sentiment_score / max_possible))

        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")

        return 0.0

class RiskManager:
    """Advanced risk management system"""

    def __init__(self, max_risk_per_trade: float = 0.02, max_total_risk: float = 0.1):
        self.max_risk_per_trade = max_risk_per_trade  # 2% per trade
        self.max_total_risk = max_total_risk  # 10% total exposure
        self.positions: List[Position] = []

    def calculate_position_size(self, account_balance: float, entry_price: float,
                              stop_loss: float) -> float:
        """Calculate optimal position size based on risk"""
        risk_amount = account_balance * self.max_risk_per_trade
        price_diff = abs(entry_price - stop_loss)

        if price_diff == 0:
            return 0

        position_size = risk_amount / price_diff
        return min(position_size, account_balance * 0.3)  # Max 30% of balance per trade

    def calculate_stop_loss(self, entry_price: float, side: str, atr: float) -> float:
        """Calculate dynamic stop loss based on ATR"""
        multiplier = 2.0  # 2x ATR
        if side == 'long':
            return entry_price - (atr * multiplier)
        else:
            return entry_price + (atr * multiplier)

    def calculate_take_profit(self, entry_price: float, stop_loss: float, side: str,
                            risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit based on risk-reward ratio"""
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio

        if side == 'long':
            return entry_price + reward
        else:
            return entry_price - reward

class DatabaseManager:
    """SQLite database management for storing trading data"""

    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        cursor = conn.cursor()

        # Create tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                symbol TEXT,
                side TEXT,
                amount REAL,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                status TEXT,
                strategy TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                symbol TEXT,
                signal TEXT,
                price REAL,
                confidence REAL,
                indicators TEXT,
                reasoning TEXT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                symbol TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
        """)

        conn.commit()
        conn.close()

    def save_signal(self, signal: TradingSignal):
        """Save trading signal to database"""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO signals (timestamp, symbol, signal, price, confidence, indicators, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.timestamp,
            'BTC/USDT',
            signal.signal,
            signal.price,
            signal.confidence,
            json.dumps(signal.indicators),
            signal.reasoning
        ))

        conn.commit()
        conn.close()

    def save_market_data(self, symbol: str, ohlcv_data: List):
        """Save market data to database"""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        cursor = conn.cursor()

        for candle in ohlcv_data:
            cursor.execute("""
                INSERT OR REPLACE INTO market_data (timestamp, symbol, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.fromtimestamp(candle[0] / 1000),
                symbol,
                candle[1], candle[2], candle[3], candle[4], candle[5]
            ))

        conn.commit()
        conn.close()

class AdvancedTradingBot:
    """Main trading bot with multiple strategies"""

    def __init__(self, config: Dict):
        logger.debug("AdvancedTradingBot.__init__ entry")
        self.config = config
        self.exchange = self.setup_exchange()
        self.technical_analyzer = TechnicalAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_manager = RiskManager(
            max_risk_per_trade=config.get('max_risk_per_trade', 0.02),
            max_total_risk=config.get('max_total_risk', 0.1)
        )
        self.db_manager = DatabaseManager()
        self.is_running = False
        self.current_position = None
        self.notification_manager = notifications.NotificationManager()

        # Trading parameters
        self.symbol = config.get('symbol', 'BTC/USDT')
        self.timeframe = config.get('timeframe', '1h')
        self.min_confidence = config.get('min_confidence', 0.7)
        logger.debug("AdvancedTradingBot.__init__ exit")

    def setup_exchange(self) -> ccxt.Exchange:
        """Setup exchange connection"""
        exchange_id = self.config.get('exchange', 'binance')
        exchange_class = getattr(ccxt, exchange_id)

        exchange = exchange_class({
            'apiKey': os.getenv('EXCHANGE_API_KEY'),
            'secret': os.getenv('EXCHANGE_SECRET'),
            'sandbox': self.config.get('sandbox', True),  # Use testnet by default
            'enableRateLimit': True,
            'verbose': False, # Disable verbose output
        })

        return exchange

    def get_market_data(self, limit: int = 100) -> pd.DataFrame:
        """Fetch and process market data"""
        logger.debug(f"AdvancedTradingBot.get_market_data entry with limit: {limit}")
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Save to database
            self.db_manager.save_market_data(self.symbol, ohlcv)
            logger.debug("AdvancedTradingBot.get_market_data exit")
            return df
        except Exception as e:
            logger.error(f"Error fetching market data: {e}", exc_info=True)
            logger.debug("AdvancedTradingBot.get_market_data exit with error")
            return pd.DataFrame()

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def generate_multi_strategy_signal(self, df: pd.DataFrame) -> TradingSignal:
        """Generate trading signal using multiple strategies"""
        logger.debug("AdvancedTradingBot.generate_multi_strategy_signal entry")
        if len(df) < 50:  # Not enough data
            logger.debug("AdvancedTradingBot.generate_multi_strategy_signal exit: Insufficient data")
            return TradingSignal(
                timestamp=datetime.now(),
                signal='HOLD',
                price=df['close'].iloc[-1],
                confidence=0.0,
                indicators={},
                reasoning="Insufficient data"
            )

        # Calculate all indicators
        indicators = self.technical_analyzer.analyze_all_indicators(df)
        current_price = df['close'].iloc[-1]

        # Get sentiment data
        fear_greed = self.sentiment_analyzer.get_fear_greed_index()
        news_sentiment = self.sentiment_analyzer.get_crypto_news_sentiment()

        # Strategy 1: Trend Following
        trend_score = self.trend_following_strategy(indicators, current_price)

        # Strategy 2: Mean Reversion
        mean_reversion_score = self.mean_reversion_strategy(indicators, current_price)

        # Strategy 3: Momentum Strategy
        momentum_score = self.momentum_strategy(indicators)

        # Strategy 4: Sentiment Strategy
        sentiment_score = self.sentiment_strategy(fear_greed, news_sentiment)

        # Combine all strategies with weights
        weights = {
            'trend': 0.3,
            'mean_reversion': 0.2,
            'momentum': 0.3,
            'sentiment': 0.2
        }

        total_score = (
            trend_score * weights['trend'] +
            mean_reversion_score * weights['mean_reversion'] +
            momentum_score * weights['momentum'] +
            sentiment_score * weights['sentiment']
        )

        # Determine signal and confidence
        if total_score > 0.3:
            signal = 'BUY'
            confidence = min(total_score, 1.0)
        elif total_score < -0.3:
            signal = 'SELL'
            confidence = min(abs(total_score), 1.0)
        else:
            signal = 'HOLD'
            confidence = 1.0 - abs(total_score)

        reasoning = f"Trend: {trend_score:.2f},\nMean Rev: {mean_reversion_score:.2f},\n" \
                   f"Momentum: {momentum_score:.2f},\nSentiment: {sentiment_score:.2f}"

        trading_signal = TradingSignal(
            timestamp=datetime.now(),
            signal=signal,
            price=current_price,
            confidence=confidence,
            indicators={
                'trend_score': trend_score,
                'mean_reversion_score': mean_reversion_score,
                'momentum_score': momentum_score,
                'sentiment_score': sentiment_score,
                'total_score': total_score,
                'fear_greed': fear_greed,
                'news_sentiment': news_sentiment
            },
            reasoning=reasoning
        )

        # Save signal to database
        self.db_manager.save_signal(trading_signal)
        logger.debug("AdvancedTradingBot.generate_multi_strategy_signal exit")
        return trading_signal

    def trend_following_strategy(self, indicators: Dict, current_price: float) -> float:
        """Trend following strategy scoring"""
        score = 0.0

        # Moving average alignment
        sma_20 = indicators['sma_20'].iloc[-1]
        sma_50 = indicators['sma_50'].iloc[-1]
        ema_12 = indicators['ema_12'].iloc[-1]
        ema_26 = indicators['ema_26'].iloc[-1]

        # Price above/below MAs
        if current_price > sma_20 > sma_50:
            score += 0.3
        elif current_price < sma_20 < sma_50:
            score -= 0.3

        # EMA crossover
        if ema_12 > ema_26:
            score += 0.2
        else:
            score -= 0.2

        # MACD trend
        macd_line = indicators['macd']['macd'].iloc[-1]
        signal_line = indicators['macd']['signal'].iloc[-1]

        if macd_line > signal_line:
            score += 0.1
        else:
            score -= 0.1

        return max(-1.0, min(1.0, score))

    def mean_reversion_strategy(self, indicators: Dict, current_price: float) -> float:
        """Mean reversion strategy scoring"""
        score = 0.0

        # RSI levels
        rsi = indicators['rsi'].iloc[-1]
        if rsi < 30:  # Oversold
            score += 0.4
        elif rsi > 70:  # Overbought
            score -= 0.4

        # Bollinger Bands
        bb_upper = indicators['bollinger']['upper'].iloc[-1]
        bb_lower = indicators['bollinger']['lower'].iloc[-1]
        bb_middle = indicators['bollinger']['middle'].iloc[-1]

        if current_price < bb_lower:  # Price below lower band
            score += 0.3
        elif current_price > bb_upper:  # Price above upper band
            score -= 0.3

        # Stochastic
        k_percent = indicators['stochastic']['k_percent'].iloc[-1]
        if k_percent < 20:  # Oversold
            score += 0.2
        elif k_percent > 80:  # Overbought
            score -= 0.2

        return max(-1.0, min(1.0, score))

    def momentum_strategy(self, indicators: Dict) -> float:
        """Momentum strategy scoring"""
        score = 0.0

        # MACD momentum
        macd_histogram = indicators['macd']['histogram'].iloc[-1]
        prev_histogram = indicators['macd']['histogram'].iloc[-2]

        if macd_histogram > prev_histogram and macd_histogram > 0:
            score += 0.3
        elif macd_histogram < prev_histogram and macd_histogram < 0:
            score -= 0.3

        # RSI momentum
        rsi = indicators['rsi'].iloc[-1]
        prev_rsi = indicators['rsi'].iloc[-2]

        if rsi > prev_rsi and rsi > 50:
            score += 0.2
        elif rsi < prev_rsi and rsi < 50:
            score -= 0.2

        # Stochastic momentum
        k_percent = indicators['stochastic']['k_percent'].iloc[-1]
        d_percent = indicators['stochastic']['d_percent'].iloc[-1]

        if k_percent > d_percent and k_percent > 50:
            score += 0.1
        elif k_percent < d_percent and k_percent < 50:
            score -= 0.1

        return max(-1.0, min(1.0, score))

    def sentiment_strategy(self, fear_greed: Optional[Dict], news_sentiment: float) -> float:
        """Sentiment-based strategy scoring"""
        score = 0.0

        # Fear & Greed Index
        if fear_greed:
            fg_value = fear_greed['value']
            if fg_value < 25:  # Extreme Fear - potential buy signal
                score += 0.3
            elif fg_value > 75:  # Extreme Greed - potential sell signal
                score -= 0.3
            elif fg_value < 45:  # Fear
                score += 0.1
            elif fg_value > 55:  # Greed
                score -= 0.1

        # News sentiment
        score += news_sentiment * 0.5  # Scale news sentiment

        return max(-1.0, min(1.0, score))

    def execute_trade(self, signal: TradingSignal) -> bool:
        """Execute trade based on signal"""
        logger.debug(f"AdvancedTradingBot.execute_trade entry with signal: {signal.signal}")
        try:
            if signal.confidence < self.min_confidence:
                logger.info(f"Signal confidence {signal.confidence:.2f} below minimum {self.min_confidence}")
                logger.debug("AdvancedTradingBot.execute_trade exit: Confidence too low")
                return False

            # Get account balance
            balance = self.exchange.fetch_balance()
            available_balance = balance['USDT']['free'] if 'USDT' in balance else 0

            if available_balance < 10:  # Minimum $10 to trade
                logger.warning("Insufficient balance to execute trade")
                logger.debug("AdvancedTradingBot.execute_trade exit: Insufficient balance")
                return False

            # Calculate position parameters
            df = self.get_market_data(50)
            atr = self.calculate_atr(df).iloc[-1]

            if signal.signal == 'BUY' and not self.current_position:
                stop_loss = self.risk_manager.calculate_stop_loss(signal.price, 'long', atr)
                take_profit = self.risk_manager.calculate_take_profit(signal.price, stop_loss, 'long')
                position_size = self.risk_manager.calculate_position_size(available_balance, signal.price, stop_loss)

                # Execute buy order (in sandbox/testnet)
                logger.info(f"Executing BUY order: {position_size:.6f} BTC at {signal.price:.2f}")
                logger.info(f"Stop Loss: {stop_loss:.2f}, Take Profit: {take_profit:.2f}")

                # In real implementation, execute actual order:
                # order = self.exchange.create_market_buy_order(self.symbol, position_size)

                self.current_position = Position(
                    symbol=self.symbol,
                    side='long',
                    amount=position_size,
                    entry_price=signal.price,
                    entry_time=datetime.now(),
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )

                return True

            elif signal.signal == 'SELL' and self.current_position and self.current_position.side == 'long':
                # Close long position
                logger.info(f"Closing LONG position: {self.current_position.amount:.6f} BTC at {signal.price:.2f}")

                # Calculate PnL
                pnl = (signal.price - self.current_position.entry_price) * self.current_position.amount
                logger.info(f"Position PnL: ${pnl:.2f}")

                # In real implementation, execute actual sell order:
                # order = self.exchange.create_market_sell_order(self.symbol, self.current_position.amount)

                self.current_position = None
                return True

        except Exception as e:
            logger.error(f"Error executing trade: {e}", exc_info=True)
            logger.debug("AdvancedTradingBot.execute_trade exit with error")
            return False

        logger.debug("AdvancedTradingBot.execute_trade exit: No trade executed")
        return False

    def monitor_positions(self):
        """Monitor open positions for stop loss/take profit"""
        logger.debug("AdvancedTradingBot.monitor_positions entry")
        if not self.current_position:
            logger.debug("AdvancedTradingBot.monitor_positions exit: No current position")
            return

        try:
            current_price = self.exchange.fetch_ticker(self.symbol)['last']

            # Check stop loss
            if (self.current_position.side == 'long' and current_price <= self.current_position.stop_loss):
                logger.info(f"Stop loss triggered at {current_price:.2f}")
                # Execute stop loss order
                self.current_position = None

            # Check take profit
            elif (self.current_position.side == 'long' and current_price >= self.current_position.take_profit):
                logger.info(f"Take profit triggered at {current_price:.2f}")
                # Execute take profit order
                self.current_position = None

            # Update current PnL
            if self.current_position:
                self.current_position.current_pnl = (current_price - self.current_position.entry_price) * self.current_position.amount

        except Exception as e:
            logger.error(f"Error monitoring positions: {e}", exc_info=True)
            logger.debug("AdvancedTradingBot.monitor_positions exit with error")

    def run_analysis_cycle(self):
        """Run one complete analysis cycle"""
        logger.debug("AdvancedTradingBot.run_analysis_cycle entry")
        try:
            logger.info("Starting analysis cycle...")

            # Get market data
            df = self.get_market_data()
            if df.empty:
                logger.warning("No market data available")
                return

            # Generate signal
            signal = self.generate_multi_strategy_signal(df)
            logger.info(f"Generated signal: {signal.signal} (confidence: {signal.confidence:.2f})")
            logger.info(f"Reasoning: {signal.reasoning}")

            # Kirim notifikasi otomatis jika sinyal BUY/SELL
            if signal.signal in ['BUY', 'SELL']:
                self.notification_manager.notify_trade_signal(
                    signal.signal,
                    signal.price,
                    signal.confidence,
                    signal.reasoning
                )

            # Execute trade if conditions met
            if signal.signal in ['BUY', 'SELL']:
                self.execute_trade(signal)

            # Monitor existing positions
            self.monitor_positions()

        except Exception as e:
            logger.error(f"Error in analysis cycle: {e}", exc_info=True)
            logger.debug("AdvancedTradingBot.run_analysis_cycle exit with error")
        logger.debug("AdvancedTradingBot.run_analysis_cycle exit")

    def start_bot(self):
        """Start the trading bot"""
        logger.info("Starting Advanced Bitcoin Trading Bot...")
        self.is_running = True

        # Initial execution
        self.run_analysis_cycle()
        self.monitor_positions()

        # Schedule analysis cycles
        schedule.every(1).hour.do(self.run_analysis_cycle)  # Run every 1 hour
        schedule.every(1).hour.do(self.monitor_positions)  # Monitor positions every 1 hour

        next_run_time = datetime.now() + timedelta(hours=1) # Initial next run time

        while self.is_running:
            schedule.run_pending()

            # Calculate time until next scheduled run_analysis_cycle
            # This assumes run_analysis_cycle is the primary scheduled task
            # and it runs every hour.
            # If schedule.next_run is not reliable for the exact next run of a specific job,
            # we might need to track it manually or adjust.

            # Find the next scheduled run for run_analysis_cycle
            # This is a simplification; schedule library doesn't directly expose next run for specific jobs easily.
            # For this implementation, we'll assume a fixed 1-hour interval from the last run.

            # Instead of relying on schedule.next_run, we'll calculate based on a fixed interval
            # and update next_run_time after each run_analysis_cycle.

            # For the purpose of this interactive countdown, we'll simulate the 1-hour interval.
            # A more robust solution would involve tracking the actual next scheduled time from the scheduler.

            remaining_seconds = (next_run_time - datetime.now()).total_seconds()

            if remaining_seconds <= 0:
                # If the scheduled time has passed, update next_run_time for the next cycle
                next_run_time = datetime.now() + timedelta(hours=1)
                remaining_seconds = (next_run_time - datetime.now()).total_seconds() # Recalculate

            for i in range(int(remaining_seconds), 0, -1):
                if not self.is_running:
                    break
                minutes, seconds = divmod(i, 60)
                hours, minutes = divmod(minutes, 60)
                print(f"\rNext analysis in: {hours:02d}:{minutes:02d}:{seconds:02d}", end="", flush=True)
                time.sleep(1)

            print("\r" + " " * 50 + "\r", end="", flush=True) # Clear the line after countdown

            # After the countdown, if the bot is still running, it means a cycle is about to start or just started.
            # We update next_run_time to reflect the next hour.
            if self.is_running:
                next_run_time = datetime.now() + timedelta(hours=1)

    def stop_bot(self):
        """Stop the trading bot"""
        logger.info("Stopping trading bot...")
        self.is_running = False

def main():
    """Main function to run the trading bot"""

    # Configuration
    load_dotenv() # Load environment variables from .env file

    config = {
        'exchange': os.getenv('EXCHANGE', 'binance'),
        'symbol': os.getenv('SYMBOL', 'BTC/USDT'),
        'timeframe': os.getenv('TIMEFRAME', '1h'),
        'sandbox': os.getenv('SANDBOX', 'True').lower() == 'true',  # Use testnet
        'max_risk_per_trade': float(os.getenv('MAX_RISK_PER_TRADE', 0.02)),  # 2% risk per trade
        'max_total_risk': float(os.getenv('MAX_TOTAL_RISK', 0.1)),  # 10% total risk
        'min_confidence': float(os.getenv('MIN_CONFIDENCE', 0.7))  # Minimum confidence to execute trades
    }

    # Log loaded configuration values
    logger.debug("Loaded configuration:")
    for key, value in config.items():
        if key not in ['EXCHANGE_API_KEY', 'EXCHANGE_SECRET', 'NEWS_API_KEY']: # Avoid logging sensitive info
            logger.debug(f"  {key}: {value}")

    # Initialize and start bot
    bot = AdvancedTradingBot(config)

    try:
        bot.start_bot()  # Jalankan bot secara berkala
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        bot.stop_bot()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        bot.stop_bot()

if __name__ == "__main__":
    main()
