"""
BTC 5min/10min 策略自动交易机器人
v1.1 — 模拟/实盘双模式, 飞书通知, 事件合约自动下单
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
import pickle
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import configparser
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import pandas as pd
import numpy as np

MODEL_FILE = "ml_model.pkl"

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
    df["rsi6"] = rsi(df["close"], 6)
    macd_l, macd_s, df["macd_hist"] = macd(df["close"])
    df["macd_line"] = macd_l
    df["macd_signal"] = macd_s
    df["vol_ma20"] = sma(df["volume"], 20)
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)

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

    # 信号 (v6.2 优化参数)
    bars_th = 3
    rsi_buy_max = 50
    rsi_sell_min = 45
    vol_min = 1.0

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

    # 震荡市场反弹
    df["range_here"] = ~df["trend_up"] & ~df["trend_dn"]
    df["range_buy"] = (
        df["range_here"] &
        (df["bars_below"] >= 2) &
        (df["close"] > df["ema8"]) &
        (df["close"] > df["open"]) &
        (df["volume"] > df["vol_ma20"] * vol_min) &
        (df["rsi14"] > df["rsi14"].shift(1)) &
        (df["rsi14"] < rsi_buy_max) &
        df["macd_rising"]
    )
    df["range_sell"] = (
        df["range_here"] &
        (df["bars_above"] >= 2) &
        (df["close"] < df["ema8"]) &
        (df["close"] < df["open"]) &
        (df["volume"] > df["vol_ma20"] * vol_min) &
        (df["rsi14"] < df["rsi14"].shift(1)) &
        (df["rsi14"] > rsi_sell_min) &
        df["macd_falling"]
    )

    # 综合入场
    df["buy_entry"] = df["pullback_buy"] | df["range_buy"]
    df["sell_entry"] = df["bounce_sell"] | df["range_sell"]

    # Pine Script 完整信号 (含多时间框架过滤, 无冷却)
    df["buy5"] = df["buy_entry"] & df["bull_5min"]
    df["sell5"] = df["sell_entry"] & ~df["bull_5min"]
    df["buy10"] = df["pullback_buy"] & df["bull_5min"] & df["bull_15min"]
    df["sell10"] = df["bounce_sell"] & ~df["bull_5min"] & ~df["bull_15min"]

    return df


def extend_ml_features(df):
    """在原 compute_indicators 结果上添加 ML 模型需要的额外特征列."""
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    o = df["open"].values
    vol_col = "volume" if "volume" in df.columns else "vol"
    v = df[vol_col].values
    n = len(df)

    df["above_ema8"] = c > df["ema8"].values

    # RSI 方向
    r14 = df["rsi14"].values
    r6 = df["rsi6"].values
    df["rsi14_rising"] = np.concatenate([[False], np.diff(r14) > 0])
    df["rsi6_rising"] = np.concatenate([[False], np.diff(r6) > 0])
    df["macd_above_signal"] = df["macd_line"].values > df["macd_signal"].values

    # Volume SMA20 (vol_sma20 in backtest)
    vol_sma20 = np.zeros(n)
    for i in range(n):
        vol_sma20[i] = np.mean(v[max(0, i - 19):i + 1])
    df["vol_sma20"] = vol_sma20
    df["vol_ratio"] = v / np.where(vol_sma20 == 0, 1, vol_sma20)

    # MACD 别名 (模型用 macd, auto_trade 用 macd_line)
    if "macd_line" in df.columns and "macd" not in df.columns:
        df["macd"] = df["macd_line"]

    # ATR (atr vs atr14 naming)
    prev_close = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))
    atr_arr = np.zeros(n)
    atr_arr[0] = tr[0]
    for i in range(1, n):
        atr_arr[i] = (atr_arr[i - 1] * 13 + tr[i]) / 14
    df["atr"] = atr_arr

    # 动量特征
    df["mom_3"] = np.concatenate([[0, 0], c[2:] - c[:-2]])
    df["mom_5"] = np.concatenate([[0] * 4, c[4:] - c[:-4]])

    # 连续 K 线计数
    bull = c > o
    bull_bars = np.zeros(n, dtype=np.int32)
    bear_bars = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        if bull[i]:
            bull_bars[i] = bull_bars[i - 1] + 1
            bear_bars[i] = 0
        else:
            bear_bars[i] = bear_bars[i - 1] + 1
            bull_bars[i] = 0
    df["bull_bars"] = bull_bars
    df["bear_bars"] = bear_bars

    # K 线实体比例
    hl = h - l
    hl = np.where(hl == 0, 1, hl)
    df["candle_body_ratio"] = (c - o) / hl

    # 价格区间
    high_5 = np.zeros(n)
    low_5 = np.zeros(n)
    for i in range(n):
        high_5[i] = np.max(h[max(0, i - 4):i + 1])
        low_5[i] = np.min(l[max(0, i - 4):i + 1])
    df["high_5"] = high_5
    df["low_5"] = low_5
    hl5 = high_5 - low_5
    hl5 = np.where(hl5 == 0, 1, hl5)
    df["price_position_5"] = (c - low_5) / hl5

    return df


# ==============================
# 2.5 ML Predictor
# ==============================

class MLPredictor:
    """加载使用 RandomForest 模型做涨跌预测."""

    def __init__(self, model_path=MODEL_FILE):
        self.model_5 = None
        self.model_10 = None
        self.features = []
        self.acc5 = 0
        self.acc10 = 0
        self.load(model_path)

    def load(self, path):
        if not os.path.exists(path):
            print(f"[ML] 模型文件 {path} 不存在, 请先训练: python backtest_btc.py train")
            return False
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.model_5 = payload["model_5"]
        self.model_10 = payload["model_10"]
        self.features = payload["features"]
        self.acc5 = payload.get("acc5", 0)
        self.acc10 = payload.get("acc10", 0)
        print(f"[ML] 模型已加载 (5min={self.acc5:.1%}, 10min={self.acc10:.1%}, {len(self.features)}特征)")
        return True

    def predict(self, df, conf_threshold=0.6):
        """
        对最新一根 K 线做预测, 返回 (pred_5: 1/0/None, prob_5, pred_10, prob_10).
        None = 置信度不足, 不产生信号.
        """
        if self.model_5 is None:
            return None, None, None, None

        missing = [c for c in self.features if c not in df.columns]
        if missing:
            print(f"[ML] 缺少特征: {missing}")
            return None, None, None, None

        last = df[self.features].iloc[-1:].values.astype(np.float64)
        if np.isnan(last).any():
            return None, None, None, None

        prob5 = self.model_5.predict_proba(last)[0][1]
        prob10 = self.model_10.predict_proba(last)[0][1]

        pred5 = 1 if prob5 >= conf_threshold else (0 if prob5 <= 1 - conf_threshold else None)
        pred10 = 1 if prob10 >= conf_threshold else (0 if prob10 <= 1 - conf_threshold else None)

        return pred5, prob5, pred10, prob10

def fetch_recent_klines(symbol="BTCUSDT", interval="1m", limit=500):
    """获取最近limit根K线"""
    import ssl
    # 处理 GFW 干扰: 创建自适应 SSL 上下文
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    urls = [
        f"https://api.binance.com/api/v3/klines",
        f"https://api1.binance.com/api/v3/klines",
        f"https://api2.binance.com/api/v3/klines",
        f"https://api3.binance.com/api/v3/klines",
    ]
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    for url in urls:
        try:
            resp = requests.get(url, params=params, timeout=10, verify=False)
            resp.raise_for_status()
            klines = resp.json()
            break
        except Exception:
            continue
    else:
        print(f"[数据] 获取失败 (所有端点)")
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
    def __init__(self, paper_mode=True, bet=50, pr_rate=0.8, notifier=None, hibt_bot=None):
        self.paper_mode = paper_mode
        self.bet = bet
        self.pr_rate = pr_rate
        self.notifier = notifier  # 飞书通知器
        self._hibt_bot = hibt_bot
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
            if latest["buy_entry"] and latest["bull_5min"]:
                signals.append(("5min", OrderSide.UP))
            elif latest["sell_entry"] and not latest["bull_5min"]:
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
        """实盘模式: 通过 Playwright 自动在 HIBT 事件合约下单"""
        direction = "up" if trade.side == OrderSide.UP else "down"
        duration = "5min" if trade.timeframe == "5min" else "10min"
        print(f"[实盘] 浏览器下单: {direction} {trade.stake}U {duration}")

        if not hasattr(self, '_hibt_bot') or self._hibt_bot is None:
            print("[实盘] HIBT bot 未初始化, 请使用 --hibt-email 和 --hibt-password 参数")
            return

        try:
            result = self._hibt_bot.place_bet(
                direction=direction,
                amount=trade.stake,
                duration=duration,
            )
            if result.success:
                print(f"[实盘] 下单成功!")
                if self.notifier:
                    self.notifier.send(f"🤖 自动下单成功: {'涨' if trade.side == OrderSide.UP else '跌'} {trade.stake}U {trade.timeframe}")
            else:
                print(f"[实盘] 下单失败: {result.message}")
                if self.notifier:
                    self.notifier.send(f"⚠️ 自动下单失败: {result.message}")
        except Exception as e:
            print(f"[实盘] 下单异常: {e}")

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

        lines = [
            f"[OK] 运行{uptime_h:.1f}h",
            f"交易{len(settled)}单 胜率{wr:.1f}%",
            f"盈亏{total_pnl:+.1f}U 挂单{len(pending)}",
        ]
        msg_print = " | ".join(lines)
        print(f"[心跳] {msg_print}")
        if self.notifier:
            msg_notify = (
                f"🤖 BTC {uptime_h:.1f}h\n"
                f"交易: {len(settled)}单 | 胜率: {wr:.1f}%\n"
                f"盈亏: {total_pnl:+.1f}U | 挂单: {len(pending)}"
            )
            self.notifier.send(msg_notify)

    def alert_error(self, err_msg: str):
        """连接异常时发飞书告警"""
        now = time.time()
        if now - self.last_error_alert < 300:  # 5分钟内不重复告警
            return
        self.last_error_alert = now
        self.was_disconnected = True

        print(f"[告警] 连接异常: {err_msg}")
        if self.notifier:
            self.notifier.send(f"⚠️ 连接异常\n{err_msg}")

    def alert_recovered(self):
        """连接恢复通知"""
        if not self.was_disconnected:
            return
        self.was_disconnected = False
        print(f"[恢复] 连接已恢复, 正常运行中")
        if self.notifier:
            self.notifier.send(f"[OK] 连接已恢复, 正常运行中")

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
# 4.5 期货数据获取
# ==============================

FEISHU_HOOK_CACHE = None  # 飞书 webhook 用于独立通知


def _fetch_futures_series(path, params, retries=3):
    hosts = ["fapi.binance.com", "fapi1.binance.com", "fapi2.binance.com", "fapi3.binance.com"]
    for host in hosts:
        for _ in range(retries):
            try:
                r = requests.get(f"https://{host}{path}", params=params, timeout=10, verify=False)
                if r.status_code == 200:
                    return r.json()
            except Exception:
                continue
    return []


def fetch_futures_data_live(symbol="BTCUSDT", n_records=3):
    """获取最新几笔期货数据, 返回 dict of latest values."""
    result = {}
    # Funding rate
    raw = _fetch_futures_series("/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1})
    if raw:
        result["funding_rate"] = float(raw[-1]["fundingRate"])
    # Open interest
    raw = _fetch_futures_series("/futures/data/openInterestHist", {"symbol": symbol, "period": "5m", "limit": 2})
    if raw and len(raw) >= 2:
        result["open_interest"] = float(raw[-1]["sumOpenInterest"])
        oi_prev = float(raw[-2]["sumOpenInterest"])
        result["oi_change"] = (result["open_interest"] - oi_prev) / oi_prev if oi_prev else 0
    # Taker ratio
    raw = _fetch_futures_series("/futures/data/takerlongshortRatio", {"symbol": symbol, "period": "5m", "limit": 1})
    if raw:
        result["taker_buy_ratio"] = float(raw[-1]["buySellRatio"])
    # LS ratio
    raw = _fetch_futures_series("/futures/data/globalLongShortAccountRatio", {"symbol": symbol, "period": "5m", "limit": 1})
    if raw:
        result["ls_ratio"] = float(raw[-1]["longShortRatio"])
    return result


def apply_futures_to_df(df, futures_dict):
    """将最新期货数据填充到 df 最后一行."""
    if not futures_dict:
        return
    for k, v in futures_dict.items():
        if k not in df.columns:
            df[k] = 0.0
        df.loc[df.index[-1], k] = v


# ==============================
# 5. 主循环
# ==============================

def main_loop(bot: BTCAutoTrader, interval_sec=1, ml_predictor: MLPredictor = None):
    print(f"[主循环] 每{interval_sec}秒检查一次...")
    bot.running = True
    processed_bars = set()
    futures_cache_time = 0
    futures_cache = {}

    print("[启动] 加载历史数据...")
    df = fetch_recent_klines(symbol=bot.symbol, interval=bot.interval, limit=200)
    if df is not None and len(df) >= 100:
        df = compute_indicators(df)
        df = extend_ml_features(df)
        for ts in df.index:
            processed_bars.add(ts)
        signals = bot.process_bar(df)
        if not signals:
            print(f"[启动] 当前无信号, 等待新K线...")
        print(f"[启动] 就绪, 最新K线: {df.index[-1]} 收盘={df.iloc[-1]['close']:.2f}")

    if ml_predictor:
        futures_cache = fetch_futures_data_live()
        futures_cache_time = time.time()

    try:
        while bot.running:
            time.sleep(interval_sec)
            bot.check_heartbeat()

            df = fetch_recent_klines(symbol=bot.symbol, interval=bot.interval, limit=200)
            if df is None:
                bot.consecutive_errors += 1
                if bot.consecutive_errors == 60:
                    bot.alert_error("币安API已断连60秒, 正在持续重试...")
                elif bot.consecutive_errors > 60 and bot.consecutive_errors % 300 == 0:
                    bot.alert_error(f"币安API仍不可用, 已断连 {bot.consecutive_errors//60} 分钟")
                time.sleep(1)
                continue

            if bot.consecutive_errors > 0:
                bot.consecutive_errors = 0
                bot.alert_recovered()
            else:
                bot.consecutive_errors = 0

            latest_bar = df.index[-1]
            if latest_bar not in processed_bars and len(df) >= 100:
                df = compute_indicators(df)
                df = extend_ml_features(df)

                signals = bot.process_bar(df)

                if ml_predictor:
                    if time.time() - futures_cache_time > 60:
                        futures_cache = fetch_futures_data_live()
                        futures_cache_time = time.time()
                    apply_futures_to_df(df, futures_cache)
                    pred5, prob5, pred10, prob10 = ml_predictor.predict(df, conf_threshold=0.6)
                    if pred5 is not None:
                        direction = "涨" if pred5 == 1 else "跌"
                        conf = max(prob5, 1 - prob5) * 100
                        print(f"  [ML] 5min {direction} (conf={conf:.0f}%)")
                        if bot.notifier:
                            bot.notifier.send(f"[ML] BTC 5min {direction} {conf:.0f}%\n价格: {df.iloc[-1]['close']:.2f}\n时间: {latest_bar.strftime('%H:%M')}")
                    if pred10 is not None:
                        direction = "涨" if pred10 == 1 else "跌"
                        conf = max(prob10, 1 - prob10) * 100
                        print(f"  [ML] 10min {direction} (conf={conf:.0f}%)")
                        if bot.notifier:
                            bot.notifier.send(f"[ML] BTC 10min {direction} {conf:.0f}%\n价格: {df.iloc[-1]['close']:.2f}\n时间: {latest_bar.strftime('%H:%M')}")

                processed_bars.add(latest_bar)
                if signals:
                    for tf, side in signals:
                        d = "涨" if side == OrderSide.UP else "跌"
                        print(f"  -> {tf} {d}")

                if len(processed_bars) > 500:
                    processed_bars = set(sorted(processed_bars)[-200:])

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
# 8. 入口
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

    # 实盘模式: 初始化浏览器自动化
    hibt_bot = None
    if not paper_mode:
        hibt_email = config["HIBT_WEB"].get("email", "")
        hibt_password = config["HIBT_WEB"].get("password", "")
        headless = config["HIBT_WEB"].getboolean("headless", False)

        if hibt_email and hibt_password:
            print("[实盘] 初始化 HIBT 事件合约浏览器自动化...")
            try:
                from hibt_event_bot import HIBTEventBot
                hibt_bot = HIBTEventBot(headless=headless)
                hibt_bot.start()
                if hibt_bot.login(hibt_email, hibt_password):
                    print("[实盘] HIBT 登录成功, 准备自动下单")
                    if notifier:
                        notifier.send("🤖 HIBT 事件合约自动下单已就绪")
                else:
                    print("[实盘] HIBT 登录失败, 仅通知模式")
                    hibt_bot = None
            except Exception as e:
                print(f"[实盘] HIBT bot 初始化失败: {e}")
                print("[实盘] 回退到仅通知模式")
                hibt_bot = None
        else:
            print("[实盘] 未配置 HIBT_WEB 账号密码, 仅通知模式")
            print("[实盘] 请在 config.ini [HIBT_WEB] 中配置 email 和 password")

        print("[实盘] 信号通知 + 浏览器自动下单")
    else:
        print("[模拟] 模式: 仅信号检测, 不实际下单")

    bot = BTCAutoTrader(paper_mode=paper_mode, bet=bet, pr_rate=pr_rate,
                        notifier=notifier, hibt_bot=hibt_bot)

    # ML 预测器
    ml_predictor = MLPredictor()
    if not ml_predictor.model_5:
        print("[ML] 模型未加载, 跳过 ML 预测")
        ml_predictor = None
    else:
        print("[ML] ML 预测已启用 (置信度阈值 60%)")

    if notifier:
        notifier.send(f"BTC自动交易 {'模拟' if paper_mode else '实盘'} 已启动!")

    try:
        main_loop(bot, ml_predictor=ml_predictor)
    finally:
        if notifier:
            bot.print_summary()
            notifier.send("🤖 BTC自动交易已停止")
        if hibt_bot:
            try:
                hibt_bot.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
