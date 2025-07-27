import os
import logging
import requests
from typing import Optional, Dict
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class Alert:
    timestamp: datetime
    level: str  # INFO, WARNING, ERROR, CRITICAL
    title: str
    message: str
    data: Optional[Dict] = None

class NotificationManager:
    """Manage various notification channels"""

    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.email_enabled = False  # Can be extended
        self.webhook_url = os.getenv('WEBHOOK_URL')

    def send_telegram(self, alert: Alert) -> bool:
        """Send alert via Telegram"""
        if not self.telegram_token or not self.telegram_chat_id:
            logger.warning("Telegram credentials not configured")
            return False

        try:
            # Format message
            emoji_map = {
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '🚨'
            }

            emoji = emoji_map.get(alert.level, '📢')
            message = f"{emoji} *{alert.title}*\n\n{alert.message}"

            if alert.data:
                message += f"\n\n*Details:*"
                for key, value in alert.data.items():
                    message += f"\n• {key}: {value}"

            message += f"\n\n_Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_"

            # Send via Telegram API
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }

            response = requests.post(url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info("Telegram notification sent successfully")
                return True
            else:
                logger.error(f"Failed to send Telegram notification: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending Telegram notification: {e}")
            return False

    def send_webhook(self, alert: Alert) -> bool:
        """Send alert via webhook"""
        if not self.webhook_url:
            return False

        try:
            payload = {
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level,
                'title': alert.title,
                'message': alert.message,
                'data': alert.data
            }

            response = requests.post(self.webhook_url, json=payload, timeout=10)

            if response.status_code == 200:
                logger.info("Webhook notification sent successfully")
                return True
            else:
                logger.error(f"Failed to send webhook notification: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
            return False

    def send_alert(self, alert: Alert):
        """Send alert through all configured channels"""
        logger.log(getattr(logging, alert.level), f"{alert.title}: {alert.message}")

        # Send via Telegram
        self.send_telegram(alert)

        # Send via webhook
        self.send_webhook(alert)

    def notify_trade_signal(self, signal_type: str, price: float, confidence: float, reasoning: str):
        """Notify about trading signal"""
        alert = Alert(
            timestamp=datetime.now(),
            level='INFO',
            title=f'Trading Signal: {signal_type}',
            message=f'Signal generated for BTC/USDT at ${price:.2f}',
            data={
                'Signal Type': signal_type,
                'Price': f'${price:.2f}',
                'Confidence': f'{confidence:.1%}',
                'Reasoning': reasoning
            }
        )
        self.send_alert(alert)

    def notify_trade_execution(self, action: str, amount: float, price: float, order_id: str = None):
        """Notify about trade execution"""
        alert = Alert(
            timestamp=datetime.now(),
            level='WARNING',
            title=f'Trade Executed: {action}',
            message=f'{action} {amount:.6f} BTC at ${price:.2f}',
            data={
                'Action': action,
                'Amount': f'{amount:.6f} BTC',
                'Price': f'${price:.2f}',
                'Order ID': order_id or 'N/A'
            }
        )
        self.send_alert(alert)

    def notify_position_update(self, position_type: str, pnl: float, price: float):
        """Notify about position updates"""
        level = 'INFO' if pnl >= 0 else 'WARNING'
        alert = Alert(
            timestamp=datetime.now(),
            level=level,
            title=f'Position Update: {position_type}',
            message=f'Current PnL: ${pnl:.2f} at ${price:.2f}',
            data={
                'Position': position_type,
                'PnL': f'${pnl:.2f}',
                'Current Price': f'${price:.2f}'
            }
        )
        self.send_alert(alert)

    def notify_error(self, error_type: str, error_message: str, context: Dict = None):
        """Notify about errors"""
        alert = Alert(
            timestamp=datetime.now(),
            level='ERROR',
            title=f'Bot Error: {error_type}',
            message=error_message,
            data=context
        )
        self.send_alert(alert)

    def notify_system_status(self, status: str, uptime: str, trades_today: int):
        """Notify about system status"""
        alert = Alert(
            timestamp=datetime.now(),
            level='INFO',
            title=f'Bot Status: {status}',
            message=f'System running for {uptime}',
            data={
                'Status': status,
                'Uptime': uptime,
                'Trades Today': trades_today
            }
        )
        self.send_alert(alert)
