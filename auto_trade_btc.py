"""
BTC 5min/10min 策略自动交易机器人
v1.0 — 模拟/实盘双模式, Telegram通知
策略: v6.2 趋势回踩策略

使用:
  python auto_trade_btc.py              # 模拟模式(默认)
  python auto_trade_btc.py --live       # 实盘模式(需要API KEY)

首次运行自动创建 config.ini, 填好API KEY后再开实盘.
"""

import sys
import os
import json
import time
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import configparser
import requests

import pandas as pd
import numpy as np

# ==============================
# 1. 策略引擎 (复用回测逻辑)
# ==============================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta.where(delta < 0, 0.0))
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()


def compute_indicators(df):
    """计算所有策略指标, 返回带有信号列的DataFrame"""
    df = df.copy()

    df["ema8"] = ema(df["close"], 8)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    _, _, df["macd_hist"] = macd(df["close"])
    df["vol_ma20"] = sma(df["volume"], 20)

    # 多时间框架
    df_5min = df.resample("5min").agg({"close": "last", "high": "max", "low": "min", "volume": "sum"})
    df_5min["ema8_5min"] = ema(df_5min["close"], 8)
    df_5min["bull_5min"] = df_5min["close"] > df_5min["ema8_5min"]
    df = df.join(df_5min[["bull_5min"]], how="left")
    df["bull_5min"] = df["bull_5min"].ffill().astype(bool)

    df_15min = df.resample("15min").agg({"close": "last"})
    df_15min["ema21_15min"] = ema(df_15min["close"], 21)
    df_15min["bull_15min"] = df_15min["close"] > df_15min["ema21_15min"]
    df = df.join(df_15min[["bull_15min"]], how="left")
    df["bull_15min"] = df["bull_15min"].ffill().astype(bool)

    # 趋势方向
    df["trend_up"] = (df["ema8"] > df["ema21"]) & (df["ema8"] > df["ema8"].shift(1))
    df["trend_dn"] = (df["ema8"] < df["ema21"]) & (df["ema8"] < df["ema8"].shift(1))

    # 回踩深度
    lookback = 3
    bars_below = pd.DataFrame(0.0, index=df.index, columns=range(lookback))
    bars_above = pd.DataFrame(0.0, index=df.index, columns=range(lookback))
    for i in range(1, lookback + 1):
        bars_below[i-1] = (df["close"].shift(i) < df["ema8"].shift(i)).astype(int)
        bars_above[i-1] = (df["close"].shift(i) > df["ema8"].shift(i)).astype(int)
    df["bars_below"] = bars_below.sum(axis=1)
    df["bars_above"] = bars_above.sum(axis=1)

    # MACD柱对齐
    df["macd_rising"] = df["macd_hist"] > df["macd_hist"].shift(1)
    df["macd_falling"] = df["macd_hist"] < df["macd_hist"].shift(1)

    # 信号 (v6.2 最优参数硬编码)
    bars_th = 3
    rsi_buy_max = 55
    rsi_sell_min = 45
    vol_min = 1.2

    df["pullback_buy"] = (
        df["trend_up"] &
        (df["bars_below"] >= bars_th) &
        (df["close"] > df["ema8"]) &
        (df["close"] > df["open"]) &
        (df["volume"] > df["vol_ma20"] * vol_min) &
        (df["rsi14"] > df["rsi14"].shift(1)) &
        (df["rsi14"] < rsi_buy_max) &
        df["macd_rising"]
    )

    df["bounce_sell"] = (
        df["trend_dn"] &
        (df["bars_above"] >= bars_th) &
        (df["close"] < df["ema8"]) &
        (df["close"] < df["open"]) &
        (df["volume"] > df["vol_ma20"] * vol_min) &
        (df["rsi14"] < df["rsi14"].shift(1)) &
        (df["rsi14"] > rsi_sell_min) &
        df["macd_falling"]
    )

    return df


# ==============================
# 2. 数据获取
# ==============================

def fetch_recent_klines(symbol="BTCUSDT", interval="1m", limit=500):
    """获取最近limit根K线"""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        klines = resp.json()
    except Exception as e:
        print(f"[数据] 获取失败: {e}")
        return None

    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df.set_index("timestamp", inplace=True)
    return df


# ==============================
# 3. 交易类型
# ==============================

class OrderSide(Enum):
    UP = "up"
    DOWN = "dn"

class OrderStatus(Enum):
    PENDING = "pending"      # 等待结算
    WON = "won"
    LOST = "lost"


@dataclass
class Trade:
    id: int
    time: datetime
    side: OrderSide          # up / down
    timeframe: str           # "5min" / "10min"
    entry_price: float
    stake: float = 50.0
    payout_rate: float = 0.8
    status: OrderStatus = OrderStatus.PENDING
    settle_price: Optional[float] = None
    pnl: float = 0.0

    def settle(self, current_price: float):
        if self.status != OrderStatus.PENDING:
            return
        self.settle_price = current_price
        if self.side == OrderSide.UP:
            won = current_price > self.entry_price
        else:
            won = current_price < self.entry_price
        if won:
            self.status = OrderStatus.WON
            self.pnl = self.stake * self.payout_rate
        else:
            self.status = OrderStatus.LOST
            self.pnl = -self.stake

    @property
    def is_settled(self):
        return self.status != OrderStatus.PENDING

    @property
    def age_seconds(self):
        return (datetime.now() - self.time).total_seconds()


# ==============================
# 4. 核心机器人
# ==============================

class BTCAutoTrader:
    def __init__(self, paper_mode=True, bet=50, pr_rate=0.8, notifier=None):
        self.paper_mode = paper_mode
        self.bet = bet
        self.pr_rate = pr_rate
        self.notifier = notifier  # 飞书通知器
        self.symbol = "BTCUSDT"
        self.interval = "1m"
        self.start_time = datetime.now()

        # 冷却
        self.cooldown_5 = 0
        self.cooldown_10 = 0
        self.cool5_bars = 10
        self.cool10_bars = 20

        # 交易记录
        self.trades: list[Trade] = []
        self.next_trade_id = 1
        self.trade_log_path = "trades.json"

        # 运行控制
        self.running = False
        self.last_bar_time = None

        # 自检
        self.last_heartbeat = time.time()
        self.heartbeat_interval = 3600  # 1小时
        self.consecutive_errors = 0
        self.last_error_alert = 0
        self.was_disconnected = False

        # 统计
        self.stats_lock = threading.Lock()

        # 加载历史交易
        self._load_trades()

        print(f"[启动] {'模拟' if paper_mode else '实盘'}交易机器人")
        print(f"[启动] 每注{bet}U, 收益率{pr_rate*100:.0f}%")
        print(f"[启动] 加载{len(self.trades)}条历史交易")

    # ---- 持久化 ----

    def _load_trades(self):
        if os.path.exists(self.trade_log_path):
            try:
                with open(self.trade_log_path) as f:
                    data = json.load(f)
                for t in data:
                    trade = Trade(**t)
                    trade.side = OrderSide(trade.side)
                    trade.status = OrderStatus(trade.status)
                    trade.time = datetime.fromisoformat(trade.time)
                    self.trades.append(trade)
                if self.trades:
                    self.next_trade_id = max(t.id for t in self.trades) + 1
            except Exception as e:
                print(f"[加载] 交易记录读取失败: {e}")

    def _save_trades(self):
        data = []
        for t in self.trades:
            d = asdict(t)
            d["side"] = t.side.value
            d["status"] = t.status.value
            d["time"] = t.time.isoformat()
            data.append(d)
        with open(self.trade_log_path, "w") as f:
            json.dump(data, f, indent=2)

    # ---- 信号检查 ----

    def _get_cooldown_text(self):
        return f"冷却5:{self.cooldown_5} 冷却10:{self.cooldown_10}"

    def process_bar(self, df: pd.DataFrame):
        """处理最新一根K线, 检查信号, 结算过期订单"""
        latest = df.iloc[-1]
        current_time = df.index[-1]
        current_price = float(latest["close"])

        # 结算到期订单
        self._settle_expired(current_price)

        # 更新冷却
        if self.cooldown_5 > 0:
            self.cooldown_5 -= 1
        if self.cooldown_10 > 0:
            self.cooldown_10 -= 1

        # 检查信号
        signals = []
        if self.cooldown_5 == 0:
            if latest["pullback_buy"] and latest["bull_5min"]:
                signals.append(("5min", OrderSide.UP))
            elif latest["bounce_sell"] and not latest["bull_5min"]:
                signals.append(("5min", OrderSide.DOWN))

        if self.cooldown_10 == 0:
            if latest["pullback_buy"] and latest["bull_5min"] and latest["bull_15min"]:
                signals.append(("10min", OrderSide.UP))
            elif latest["bounce_sell"] and not latest["bull_5min"] and not latest["bull_15min"]:
                signals.append(("10min", OrderSide.DOWN))

        # 执行信号
        for tf, side in signals:
            self._execute_signal(tf, side, current_time, current_price)

        self.last_bar_time = current_time
        return signals

    def _execute_signal(self, timeframe, side, time, price):
        """执行交易信号"""
        trade = Trade(
            id=self.next_trade_id,
            time=time,
            side=side,
            timeframe=timeframe,
            entry_price=price,
            stake=self.bet,
            payout_rate=self.pr_rate,
        )
        self.next_trade_id += 1
        self.trades.append(trade)

        # 冷却
        if timeframe == "5min":
            self.cooldown_5 = self.cool5_bars
        else:
            self.cooldown_10 = self.cool10_bars

        # 输出
        side_text = "做多" if side == OrderSide.UP else "做空"
        mode_text = "模拟" if self.paper_mode else "实盘"
        direction = "涨" if side == OrderSide.UP else "跌"
        print(f"[{mode_text}] #{trade.id} {timeframe} {side_text} @ {price:.2f} | {self._get_cooldown_text()}")

        # Telegram通知
        if self.notifier:
            msg = (
                f"🤖 BTC {timeframe} {direction}\n"
                f"价格: {price:.2f}\n"
                f"金额: {self.bet}U | 收益: {self.pr_rate*100:.0f}%\n"
                f"时间: {time.strftime('%H:%M:%S')}"
            )
            self.notifier.send(msg)

        # 如果是实盘, 调用API下单
        if not self.paper_mode:
            self._place_live_order(trade)

        # 持久化
        self._save_trades()

    def _place_live_order(self, trade: Trade):
        """实盘下单 (需要配置API KEY)"""
        # TODO: 接入币安事件合约API
        print(f"[实盘] 下单 #{trade.id}: {trade.timeframe} {'涨' if trade.side == OrderSide.UP else '跌'} 金额={trade.stake}U")

    def _settle_expired(self, current_price: float):
        """结算所有到期的挂单"""
        settled_count = 0
        for trade in self.trades:
            if trade.status != OrderStatus.PENDING:
                continue

            elapsed = (datetime.now() - trade.time).total_seconds()
            need_seconds = 300 if trade.timeframe == "5min" else 600  # 5min/10min

            if elapsed >= need_seconds:
                trade.settle(current_price)
                settled_count += 1
                result = "✅" if trade.status == OrderStatus.WON else "❌"
                pnl_text = f"+{trade.pnl:.1f}U" if trade.pnl >= 0 else f"{trade.pnl:.1f}U"
                print(f"  {result} #{trade.id} {trade.timeframe} 结算: "
                      f"{'胜' if trade.status == OrderStatus.WON else '败'} "
                      f"入场{trade.entry_price:.2f}→出场{trade.settle_price:.2f} "
                      f"盈亏={pnl_text}")

                # Telegram结算通知
                if self.notifier:
                    emoji = "✅" if trade.status == OrderStatus.WON else "❌"
                    direction = "涨" if trade.side == OrderSide.UP else "跌"
                    self.notifier.send(
                        f"{emoji} BTC {trade.timeframe} {direction} 结算\n"
                        f"结果: {'获胜' if trade.status == OrderStatus.WON else '亏损'}\n"
                        f"盈亏: {pnl_text}\n"
                        f"入场: {trade.entry_price:.2f} → 出场: {trade.settle_price:.2f}"
                    )

        if settled_count > 0:
            self._save_trades()
            self._print_stats()

    # ---- 统计 ----

    def _print_stats(self):
        with self.stats_lock:
            settled = [t for t in self.trades if t.is_settled]
            pending = [t for t in self.trades if not t.is_settled]
            wins = sum(1 for t in settled if t.status == OrderStatus.WON)
            losses = len(settled) - wins
            total_pnl = sum(t.pnl for t in settled)

        if not settled:
            return

        wr = wins / len(settled) * 100
        print(f"  ── 统计: {wins}胜/{losses}败 胜率{wr:.1f}% 盈亏{total_pnl:+.1f}U "
              f"挂单中:{len(pending)}")

    def check_heartbeat(self):
        """定时心跳: 每小时发一次运行状态到飞书"""
        now = time.time()
        if now - self.last_heartbeat < self.heartbeat_interval:
            return
        self.last_heartbeat = now

        settled = [t for t in self.trades if t.is_settled]
        pending = [t for t in self.trades if not t.is_settled]
        wins = sum(1 for t in settled if t.status == OrderStatus.WON)
        losses = len(settled) - wins
        total_pnl = sum(t.pnl for t in settled)
        uptime_h = (datetime.now() - self.start_time).total_seconds() / 3600
        wr = wins / len(settled) * 100 if settled else 0

        msg = (
            f"💚 运行正常 | 已运行{uptime_h:.1f}h\n"
            f"交易: {len(settled)}单 | 胜率: {wr:.1f}%\n"
            f"盈亏: {total_pnl:+.1f}U | 挂单: {len(pending)}"
        )
        print(f"[心跳] {msg}")
        if self.notifier:
            self.notifier.send(msg)

    def alert_error(self, err_msg: str):
        """连接异常时发飞书告警"""
        now = time.time()
        if now - self.last_error_alert < 300:  # 5分钟内不重复告警
            return
        self.last_error_alert = now
        self.was_disconnected = True

        msg = f"⚠️ 连接异常\n{err_msg}"
        print(f"[告警] {msg}")
        if self.notifier:
            self.notifier.send(msg)

    def alert_recovered(self):
        """连接恢复通知"""
        if not self.was_disconnected:
            return
        self.was_disconnected = False
        msg = "✅ 连接已恢复, 正常运行中"
        print(f"[恢复] {msg}")
        if self.notifier:
            self.notifier.send(msg)

    def print_summary(self):
        """打印完整统计"""
        settled = [t for t in self.trades if t.is_settled]
        pending = [t for t in self.trades if not t.is_settled]
        wins = sum(1 for t in settled if t.status == OrderStatus.WON)
        losses = len(settled) - wins
        pnl5 = sum(t.pnl for t in settled if t.timeframe == "5min")
        pnl10 = sum(t.pnl for t in settled if t.timeframe == "10min")
        total_pnl = sum(t.pnl for t in settled)
        wr = wins / len(settled) * 100 if settled else 0

        print("\n" + "=" * 50)
        print("  BTC 自动交易 运行报告")
        print("=" * 50)
        print(f"  模式: {'模拟' if self.paper_mode else '实盘'}")
        print(f"  运行时间: 待统计")
        print(f"  ── ── ── ── ──")
        print(f"  总交易: {len(settled)} ({len(pending)}挂单中)")
        print(f"  胜率:   {wr:.1f}%")
        print(f"  净盈亏:  {total_pnl:+.1f}U (5min={pnl5:+.1f} 10min={pnl10:+.1f})")
        wr5 = sum(1 for t in settled if t.timeframe == "5min" and t.status == OrderStatus.WON)
        t5 = sum(1 for t in settled if t.timeframe == "5min")
        wr10 = sum(1 for t in settled if t.timeframe == "10min" and t.status == OrderStatus.WON)
        t10 = sum(1 for t in settled if t.timeframe == "10min")
        if t5 > 0:
            print(f"  5min:   {t5}次 胜率{wr5/t5*100:.1f}%")
        if t10 > 0:
            print(f"  10min:  {t10}次 胜率{wr10/t10*100:.1f}%")
        print("=" * 50)


# ==============================
# 5. 主循环
# ==============================

def main_loop(bot: BTCAutoTrader, interval_sec=1):
    """
    主循环: 每interval_sec秒拉取K线, 检查新信号
    - 策略基于收盘价, 每根1min K线收盘时检查信号
    """
    print(f"[主循环] 每{interval_sec}秒检查一次...")
    bot.running = True

    processed_bars = set()

    # 启动时先处理已有的数据
    print("[启动] 加载历史数据...")
    df = fetch_recent_klines(symbol=bot.symbol, interval=bot.interval, limit=200)
    if df is not None and len(df) >= 100:
        df = compute_indicators(df)
        for ts in df.index:
            processed_bars.add(ts)
        signals = bot.process_bar(df)
        if not signals:
            print(f"[启动] 当前无信号, 等待新K线...")
        print(f"[启动] 就绪, 最新K线: {df.index[-1]} 收盘={df.iloc[-1]['close']:.2f}")

    try:
        while bot.running:
            time.sleep(interval_sec)

            # 心跳
            bot.check_heartbeat()

            df = fetch_recent_klines(symbol=bot.symbol, interval=bot.interval, limit=200)
            if df is None:
                bot.consecutive_errors += 1
                if bot.consecutive_errors == 1:
                    bot.alert_error("无法连接币安API, 正在重试...")
                elif bot.consecutive_errors >= 30:
                    if bot.consecutive_errors % 30 == 0:  # 每30次(30秒)提醒一次
                        bot.alert_error(f"已断连 {bot.consecutive_errors} 秒")
                time.sleep(5)
                continue

            # 连接恢复
            if bot.consecutive_errors > 0:
                bot.consecutive_errors = 0
                bot.alert_recovered()
            else:
                bot.consecutive_errors = 0

            # 检查是否有新K线
            latest_bar = df.index[-1]
            if latest_bar not in processed_bars:
                if len(df) >= 100:
                    df = compute_indicators(df)
                    signals = bot.process_bar(df)
                    processed_bars.add(latest_bar)
                    if signals:
                        for tf, side in signals:
                            d = "涨" if side == OrderSide.UP else "跌"
                            print(f"  → 信号: {tf} {d}")

                # 清理已处理列表, 只保留最近200个
                if len(processed_bars) > 500:
                    processed_bars = set(sorted(processed_bars)[-200:])
            else:
                # 安静模式下不打印, 调试时取消注释
                pass

    except KeyboardInterrupt:
        print("\n[停止] 用户中断")
    finally:
        bot.running = False
        bot.print_summary()


# ==============================
# 6. 通知 (Telegram / 飞书)
# ==============================

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    def send(self, text: str):
        try:
            requests.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=5
            )
        except Exception as e:
            print(f"[Telegram] 发送失败: {e}")


class FeishuNotifier:
    """飞书群机器人通知"""
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    def send(self, text: str):
        try:
            requests.post(
                self.webhook_url,
                json={"msg_type": "text", "content": {"text": text}},
                timeout=5
            )
        except Exception as e:
            print(f"[飞书] 发送失败: {e}")


# ==============================
# 7. 入口
# ==============================

def load_config():
    """加载/创建配置"""
    config = configparser.ConfigParser()
    if not os.path.exists("config.ini"):
        config["BINANCE"] = {
            "api_key": "",
            "api_secret": "",
        }
        config["TELEGRAM"] = {
            "bot_token": "",
            "chat_id": "",
        }
        config["FEISHU"] = {
            "webhook_url": "",
        }
        config["TRADING"] = {
            "bet": "50",
            "payout_rate": "0.8",
        }
        with open("config.ini", "w") as f:
            config.write(f)
        print("[配置] 已创建 config.ini, 请填写API信息后重启")
        return config
    config.read("config.ini")
    return config


def main():
    config = load_config()

    # 命令行参数
    paper_mode = "--live" not in sys.argv

    bet = float(config["TRADING"].get("bet", 50))
    pr_rate = float(config["TRADING"].get("payout_rate", 0.8))

    # 实盘模式检查
    if not paper_mode:
        api_key = config["BINANCE"].get("api_key", "")
        api_secret = config["BINANCE"].get("api_secret", "")
        if not api_key or not api_secret:
            print("[错误] 实盘模式需要填写 config.ini 中的 API KEY")
            print("  1. 在币安创建API (开通交易权限)")
            print("  2. 填入 config.ini [BINANCE] 字段")
            return

    # 可选: 通知 (优先飞书, 其次Telegram)
    feishu_url = config["FEISHU"].get("webhook_url", "")
    tg_token = config["TELEGRAM"].get("bot_token", "")
    tg_chat = config["TELEGRAM"].get("chat_id", "")

    if feishu_url:
        notifier = FeishuNotifier(feishu_url)
        print("[通知] 使用飞书群机器人")
    elif tg_token and tg_chat:
        notifier = TelegramNotifier(tg_token, tg_chat)
        print("[通知] 使用Telegram")
    else:
        notifier = None
        print("[通知] 未配置, 仅终端输出")

    bot = BTCAutoTrader(paper_mode=paper_mode, bet=bet, pr_rate=pr_rate, notifier=notifier)

    if notifier:
        notifier.send(f"BTC自动交易 {'模拟' if paper_mode else '实盘'} 已启动!")

    try:
        main_loop(bot)
    finally:
        if notifier:
            bot.print_summary()
            notifier.send("🤖 BTC自动交易已停止")


if __name__ == "__main__":
    main()
