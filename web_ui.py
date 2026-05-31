"""
BTC 策略实时监控 Web UI + 自动交易
Flask + lightweight-charts 实时显示K线、指标、信号
"""
import sys
import os
import time
import json
import threading
import configparser
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

import pandas as pd
import numpy as np
from flask import Flask, render_template_string, Response, jsonify, request

sys.path.insert(0, ".")
from auto_trade_btc import fetch_recent_klines, compute_indicators, extend_ml_features, MLPredictor, fetch_futures_data_live, apply_futures_to_df

app = Flask(__name__)

# ====== 飞书通知 ======
class FeishuNotifier:
    def __init__(self, webhook_url: str):
        self.url = webhook_url
    def send(self, text: str):
        try:
            import requests
            requests.post(self.url, json={"msg_type": "text", "content": {"text": text}}, timeout=5)
        except Exception as e:
            print(f"[飞书] 发送失败: {e}")

# 加载配置
cfg = configparser.ConfigParser()
cfg.read("config.ini")
notifier = FeishuNotifier(cfg["FEISHU"]["webhook_url"]) if cfg["FEISHU"].get("webhook_url") else None
BET = float(cfg["TRADING"].get("bet", 50))
PR_RATE = float(cfg["TRADING"].get("payout_rate", 0.8))

# ====== 交易状态 ======
class OrderSide(Enum):
    UP = "up"
    DOWN = "dn"

class OrderStatus(Enum):
    PENDING = "pending"
    WON = "won"
    LOST = "lost"

@dataclass
class Trade:
    id: int
    time: str
    price: float
    side: str
    timeframe: str
    status: str = "pending"
    pnl: float = 0.0
    is_manual: bool = False

TIMEFRAMES = {"5min": 300, "10min": 600, "30min": 1800}
pending_trades = []  # {time, price, side, timeframe, is_manual, amount, id, tf_label}
trade_history = []
trade_id = 0
trade_lock = threading.Lock()
DEFAULT_BET = float(cfg["TRADING"].get("bet", 10)) if cfg.has_section("TRADING") else 10

# 冷却 (仅自动信号)
cooldown_5 = 0
cooldown_10 = 0
last_signal_bars = set()

# 今日统计
today_date = datetime.now().date()
today_stats = {"5min_w": 0, "5min_l": 0, "10min_w": 0, "10min_l": 0, "30min_w": 0, "30min_l": 0}

# 资金管理 (持久化到 cap.json)
CAP_FILE = "cap.json"
CAPITAL = 1000.0
if os.path.exists(CAP_FILE):
    try:
        with open(CAP_FILE) as f:
            CAPITAL = json.load(f).get("capital", 1000.0)
    except: pass

def save_capital():
    with open(CAP_FILE, "w") as f:
        json.dump({"capital": round(CAPITAL, 1)}, f)

# K线手动单标记
manual_markers = []  # [{time: ts, side, price, amount, id}]

# ====== 数据缓存 ======
cache = {"df": None, "updated_at": None, "lock": threading.Lock(), "backfilled": False}

# ML 预测模型
ml_predictor = MLPredictor()
ml_result = {"pred5": None, "prob5": None, "pred10": None, "prob10": None, "conf5": 0, "conf10": 0}
futures_cache = {"funding_rate": 0.0, "open_interest": 0.0, "oi_change": 0.0, "taker_buy_ratio": 0.0, "ls_ratio": 0.0}
futures_cache_time = 0


def backfill_history(df):
    """启动时扫描历史K线, 统计过去信号和结算数据"""
    global today_stats, trade_id
    if "buy5" not in df.columns:
        return
    print("[回扫] 开始统计历史信号...")
    count_5 = count_10 = 0
    w5 = l5 = w10 = l10 = 0
    for i in range(len(df) - 15):
        row = df.iloc[i]
        entry_price = float(row["close"])
        if row["buy5"]:
            settle_idx = i + 6
            if settle_idx < len(df):
                settle_close = float(df.iloc[settle_idx - 1]["close"])
                won = settle_close > entry_price
                if won: w5 += 1
                else: l5 += 1
                count_5 += 1
        elif row["sell5"]:
            settle_idx = i + 6
            if settle_idx < len(df):
                settle_close = float(df.iloc[settle_idx - 1]["close"])
                won = settle_close < entry_price
                if won: w5 += 1
                else: l5 += 1
                count_5 += 1
        if row["buy10"]:
            settle_idx = i + 11
            if settle_idx < len(df):
                settle_close = float(df.iloc[settle_idx - 1]["close"])
                won = settle_close > entry_price
                if won: w10 += 1
                else: l10 += 1
                count_10 += 1
        elif row["sell10"]:
            settle_idx = i + 11
            if settle_idx < len(df):
                settle_close = float(df.iloc[settle_idx - 1]["close"])
                won = settle_close < entry_price
                if won: w10 += 1
                else: l10 += 1
                count_10 += 1
    with trade_lock:
        trade_id += count_5 + count_10
        for _ in range(w5):
            trade_history.append(Trade(id=0, time="", price=0, side="up", timeframe="5min", status="won", pnl=BET*PR_RATE))
        for _ in range(l5):
            trade_history.append(Trade(id=0, time="", price=0, side="dn", timeframe="5min", status="lost", pnl=-BET))
        for _ in range(w10):
            trade_history.append(Trade(id=0, time="", price=0, side="up", timeframe="10min", status="won", pnl=BET*PR_RATE))
        for _ in range(l10):
            trade_history.append(Trade(id=0, time="", price=0, side="dn", timeframe="10min", status="lost", pnl=-BET))
    wr5 = round(w5 / (w5 + l5) * 100, 1) if (w5 + l5) > 0 else 0
    wr10 = round(w10 / (w10 + l10) * 100, 1) if (w10 + l10) > 0 else 0
    print(f"[回扫] 5min: {w5+l5}单 胜率{wr5}% | 10min: {w10+l10}单 胜率{wr10}% | 总计: {count_5+count_10}单")
    cache["backfilled"] = True


def update_cache():
    """后台线程：每5秒刷新一次数据 + 检测信号 + 结算"""
    global cooldown_5, cooldown_10, trade_id, pending_trades, today_stats, today_date, ml_result, futures_cache, futures_cache_time

    while True:
        now_date = datetime.now().date()
        if now_date != today_date:
            today_date = now_date
            today_stats = {"5min_w": 0, "5min_l": 0, "10min_w": 0, "10min_l": 0, "30min_w": 0, "30min_l": 0}

        try:
            df = fetch_recent_klines(limit=500)
            if df is not None and len(df) > 100:
                df = compute_indicators(df)
                df = extend_ml_features(df)

                # ML 预测
                if ml_predictor.model_5 is not None:
                    if time.time() - futures_cache_time > 60:
                        try:
                            futures_cache = fetch_futures_data_live()
                        except Exception:
                            pass
                        futures_cache_time = time.time()
                    apply_futures_to_df(df, futures_cache)
                    try:
                        pred5, prob5, pred10, prob10 = ml_predictor.predict(df, conf_threshold=0.6)
                        ml_result["pred5"] = pred5
                        ml_result["prob5"] = prob5
                        ml_result["pred10"] = pred10
                        ml_result["prob10"] = prob10
                        ml_result["conf5"] = round(max(prob5, 1 - prob5) * 100, 1) if prob5 is not None else 0
                        ml_result["conf10"] = round(max(prob10, 1 - prob10) * 100, 1) if prob10 is not None else 0
                    except Exception as e:
                        print(f"[ML] 预测失败: {e}")

                latest = df.iloc[-1]
                current_price = float(latest["close"])
                current_bar = df.index[-1]

                with cache["lock"]:
                    was_backfilled = cache["backfilled"]
                    cache["df"] = df
                    cache["updated_at"] = datetime.now()

                if not was_backfilled:
                    backfill_history(df)

                # 更新冷却
                if cooldown_5 > 0: cooldown_5 -= 1
                if cooldown_10 > 0: cooldown_10 -= 1

                # 结算所有挂单
                settled = []
                for t in pending_trades:
                    seconds_needed = TIMEFRAMES.get(t["timeframe"], 300)
                    elapsed = (current_bar - t["time"]).total_seconds()
                    if elapsed >= seconds_needed:
                        won = (t["side"] == "up" and current_price > t["price"]) or \
                              (t["side"] == "dn" and current_price < t["price"])
                        amt = t.get("amount", BET)
                        pnl = amt * PR_RATE if won else -amt
                        is_manual = t.get("is_manual", False)
                        src = "手动" if is_manual else "信号"
                        direction = "涨" if t["side"] == "up" else "跌"
                        result = "✅" if won else "❌"
                        print(f"[结算] {src} #{t.get('id','?')} {t['timeframe']}{direction} {result} | 入场={t['price']:.0f} 出场={current_price:.0f} PnL={pnl:+.0f}U")
                        with trade_lock:
                            tid = t.get("id") or (trade_id + 1)
                            if not t.get("id"):
                                trade_id += 1
                            trade_history.append(Trade(
                                id=tid, time=t["time"].strftime("%H:%M:%S"), price=t["price"],
                                side=t["side"], timeframe=t["timeframe"],
                                status="won" if won else "lost", pnl=pnl, is_manual=is_manual
                            ))
                        tf_key = t["timeframe"].replace("min", "") + "min"
                        if won: today_stats[tf_key + "_w"] += 1
                        else: today_stats[tf_key + "_l"] += 1
                        # 更新总资金
                        global CAPITAL
                        CAPITAL += pnl
                        save_capital()
                        settled.append(t)
                for t in settled:
                    pending_trades.remove(t)

                # 检测新信号 (每根bar只检测一次)
                bar_key = current_bar.strftime("%Y%m%d%H%M")
                if bar_key not in last_signal_bars:
                    last_signal_bars.add(bar_key)
                    # 清理过旧的bar key
                    if len(last_signal_bars) > 100:
                        last_signal_bars.clear()

                    signals = []
                    if cooldown_5 == 0:
                        if latest["buy_entry"] and latest["bull_5min"]:
                            signals.append(("5min", "up"))
                        elif latest["sell_entry"] and not latest["bull_5min"]:
                            signals.append(("5min", "dn"))

                    if cooldown_10 == 0:
                        if latest["pullback_buy"] and latest["bull_5min"] and latest["bull_15min"]:
                            signals.append(("10min", "up"))
                        elif latest["bounce_sell"] and not latest["bull_5min"] and not latest["bull_15min"]:
                            signals.append(("10min", "dn"))

                    for tf, side in signals:
                        direction = "涨" if side == "up" else "跌"
                        print(f"[信号] {tf}做{direction} @ {current_price:.0f}")
                        if notifier:
                            notifier.send(f"🚨 BTC {tf}做{direction}\n价格: {current_price:.0f}\n金额: {BET}U | 收益: {PR_RATE*100:.0f}%")
                        with trade_lock:
                            trade_id += 1
                            trade_history.append(Trade(id=trade_id, time=current_bar.strftime("%H:%M"), price=current_price, side=side, timeframe=tf, status="pending"))

                        pt = {"time": current_bar, "price": current_price, "side": side, "timeframe": tf, "is_manual": False, "placed_at": time.time()}
                        pending_trades.append(pt)
                        if tf == "5min":
                            cooldown_5 = 10
                        else:
                            cooldown_10 = 20

        except Exception as e:
            print(f"[缓存] 更新失败: {e}")
        time.sleep(3)


# 启动后台刷新线程
t = threading.Thread(target=update_cache, daemon=True)
t.start()


def df_to_json(df):
    """将DataFrame转为前端需要的JSON格式"""
    records = []
    for idx, row in df.iterrows():
        ts = int(idx.timestamp())
        records.append({
            "time": ts,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "ema8": float(row["ema8"]) if not pd.isna(row["ema8"]) else None,
            "ema21": float(row["ema21"]) if not pd.isna(row["ema21"]) else None,
            "ema50": float(row["ema50"]) if not pd.isna(row["ema50"]) else None,
            "rsi": float(row["rsi14"]) if not pd.isna(row["rsi14"]) else None,
            "rsi6": float(row["rsi6"]) if not pd.isna(row["rsi6"]) else None,
            "macd_hist": float(row["macd_hist"]) if not pd.isna(row["macd_hist"]) else None,
            "macd_line": float(row["macd_line"]) if not pd.isna(row["macd_line"]) else None,
            "macd_signal": float(row["macd_signal"]) if not pd.isna(row["macd_signal"]) else None,
            "vol_ma20": float(row["vol_ma20"]) if not pd.isna(row["vol_ma20"]) else None,
            "atr14": float(row["atr14"]) if not pd.isna(row["atr14"]) else None,
            "buy_entry": bool(row["buy_entry"]),
            "sell_entry": bool(row["sell_entry"]),
            "buy5": bool(row["buy5"]),
            "sell5": bool(row["sell5"]),
            "buy10": bool(row["buy10"]),
            "sell10": bool(row["sell10"]),
            "pullback_buy": bool(row["pullback_buy"]),
            "bounce_sell": bool(row["bounce_sell"]),
            "range_buy": bool(row["range_buy"]),
            "range_sell": bool(row["range_sell"]),
            "trend_up": bool(row["trend_up"]),
            "trend_dn": bool(row["trend_dn"]),
            "bull_5min": bool(row["bull_5min"]),
            "bull_15min": bool(row["bull_15min"]),
            "bars_below": int(row["bars_below"]),
            "bars_above": int(row["bars_above"]),
        })
    return records


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/data")
def api_data():
    """返回最新数据 (JSON)"""
    with cache["lock"]:
        df = cache["df"]
        if df is None:
            return jsonify({"error": "数据加载中"})
        records = df_to_json(df)
        latest = records[-1] if records else {}
        # 信号状态
        latest_signal = "等待"
        if latest.get("buy_entry"):
            latest_signal = "做多"
        elif latest.get("sell_entry"):
            latest_signal = "做空"

        # 趋势方向
        trend = "上升" if latest.get("trend_up") else ("下降" if latest.get("trend_dn") else "震荡")

        # 交易统计
        with trade_lock:
            settled = [t for t in trade_history if t.status != "pending"]
            wins = sum(1 for t in settled if t.status == "won")
            total = len(settled)
            wr = wins / total * 100 if total > 0 else 0
            pnl5 = sum(t.pnl for t in settled if t.timeframe == "5min")
            pnl10 = sum(t.pnl for t in settled if t.timeframe == "10min")
            trade_list = [asdict(t) for t in trade_history[-50:]]

            # 历史统计 (5min)
            h5 = [t for t in settled if t.timeframe == "5min"]
            h5_w = sum(1 for t in h5 if t.status == "won")
            h5_t = len(h5)
            h5_wr = round(h5_w / h5_t * 100, 1) if h5_t > 0 else 0
            h5_pnl = sum(t.pnl for t in h5)
            # 历史统计 (10min)
            h10 = [t for t in settled if t.timeframe == "10min"]
            h10_w = sum(1 for t in h10 if t.status == "won")
            h10_t = len(h10)
            h10_wr = round(h10_w / h10_t * 100, 1) if h10_t > 0 else 0
            h10_pnl = sum(t.pnl for t in h10)

        # 今日统计
        tw5, tl5, tw10, tl10, tw30, tl30 = (
            today_stats["5min_w"], today_stats["5min_l"],
            today_stats["10min_w"], today_stats["10min_l"],
            today_stats["30min_w"], today_stats["30min_l"]
        )
        today_t5, today_t10, today_t30 = tw5 + tl5, tw10 + tl10, tw30 + tl30
        today_wr5 = round(tw5 / today_t5 * 100, 1) if today_t5 > 0 else 0
        today_wr10 = round(tw10 / today_t10 * 100, 1) if today_t10 > 0 else 0
        today_wr30 = round(tw30 / today_t30 * 100, 1) if today_t30 > 0 else 0
        today_n5 = tw5 * (BET * PR_RATE) - tl5 * BET
        today_n10 = tw10 * (BET * PR_RATE) - tl10 * BET
        today_n30 = tw30 * (BET * PR_RATE) - tl30 * BET

        return jsonify({
            "records": records,
            "latest": latest,
            "signal": latest_signal,
            "trend": trend,
            "rsi": latest.get("rsi"),
            "updated_at": cache["updated_at"].strftime("%H:%M:%S") if cache["updated_at"] else "",
            "cooldown_5": cooldown_5,
            "cooldown_10": cooldown_10,
            "stats": {
                "total": total,
                "wins": wins,
                "losses": total - wins,
                "win_rate": round(wr, 1),
                "pnl5": round(pnl5, 1),
                "pnl10": round(pnl10, 1),
                "total_pnl": round(pnl5 + pnl10, 1),
            },
            "table": {
                "h5_cnt": h5_t, "h5_wr": h5_wr, "h5_pnl": round(h5_pnl, 1),
                "h10_cnt": h10_t, "h10_wr": h10_wr, "h10_pnl": round(h10_pnl, 1),
                "total_cnt": h5_t + h10_t,
                "total_wr": round((h5_w + h10_w) / (h5_t + h10_t) * 100, 1) if (h5_t + h10_t) > 0 else 0,
                "total_pnl": round(h5_pnl + h10_pnl, 1),
                "today_cnt": today_t5 + today_t10 + today_t30,
                "today_wr5": today_wr5, "today_wr10": today_wr10, "today_wr30": today_wr30,
                "today_pnl": round(today_n5 + today_n10 + today_n30, 1),
                "today_detail_5": str(tw5) + "胜" + str(tl5) + "负",
                "today_detail_10": str(tw10) + "胜" + str(tl10) + "负",
                "today_detail_30": str(tw30) + "胜" + str(tl30) + "负",
                "today_total_detail": str(tw5 + tw10 + tw30) + "胜" + str(tl5 + tl10 + tl30) + "负",
                "trend": "上升" if latest.get("trend_up") else ("下降" if latest.get("trend_dn") else "震荡"),
                "rsi_val": round(latest.get("rsi", 0), 1) if latest.get("rsi") else 0,
                "bull5": latest.get("bull_5min", False),
                "bull15": latest.get("bull_15min", False),
                "signal_text": latest_signal,
            },
            "ml": ml_result,
            "trades": trade_list[::-1],
            "capital": round(CAPITAL, 1),
            "manual_markers": manual_markers[-100:],
            "active_bets": [
                {
                    "id": t.get("id", 0),
                    "time": t["time"].strftime("%H:%M:%S"),
                    "price": round(t["price"], 1),
                    "side": t["side"],
                    "timeframe": t["timeframe"],
                    "amount": t.get("amount", BET),
                    "remaining": max(0, int(TIMEFRAMES[t["timeframe"]] - (time.time() - t.get("placed_at", time.time())))),
                }
                for t in pending_trades if not t.get("settled")
            ],
            "settled_bets": [
                {
                    "id": t.id, "time": t.time, "price": round(t.price, 1),
                    "side": t.side, "timeframe": t.timeframe,
                    "status": t.status, "pnl": round(t.pnl, 1),
                }
                for t in trade_history[-50:] if t.is_manual
            ][::-1],
        })


@app.route("/api/stream")
def stream():
    """SSE 实时推送"""
    def generate():
        while True:
            with cache["lock"]:
                df = cache["df"]
                if df is not None:
                    records = df_to_json(df)
                    latest = records[-1] if records else {}
                    with trade_lock:
                        settled = [t for t in trade_history if t.status != "pending"]
                        wr = sum(1 for t in settled if t.status == "won") / len(settled) * 100 if settled else 0
                        total_pnl_val = sum(t.pnl for t in settled)
                        # table data
                        h5 = [t for t in settled if t.timeframe == "5min"]
                        h5_w = sum(1 for t in h5 if t.status == "won"); h5_t = len(h5); h5_wr = round(h5_w/h5_t*100,1) if h5_t>0 else 0
                        h10 = [t for t in settled if t.timeframe == "10min"]
                        h10_w = sum(1 for t in h10 if t.status == "won"); h10_t = len(h10); h10_wr = round(h10_w/h10_t*100,1) if h10_t>0 else 0
                        total_cnt = h5_t + h10_t
                        total_wr = round((h5_w+h10_w)/total_cnt*100,1) if total_cnt>0 else 0
                    tw5, tl5, tw10, tl10, tw30, tl30 = (
                        today_stats["5min_w"], today_stats["5min_l"],
                        today_stats["10min_w"], today_stats["10min_l"],
                        today_stats["30min_w"], today_stats["30min_l"]
                    )
                    today_t5, today_t10, today_t30 = tw5 + tl5, tw10 + tl10, tw30 + tl30
                    today_wr5 = round(tw5/today_t5*100,1) if today_t5>0 else 0
                    today_wr10 = round(tw10/today_t10*100,1) if today_t10>0 else 0
                    today_wr30 = round(tw30/today_t30*100,1) if today_t30>0 else 0
                    today_n5 = tw5*(BET*PR_RATE)-tl5*BET
                    today_n10 = tw10*(BET*PR_RATE)-tl10*BET
                    today_n30 = tw30*(BET*PR_RATE)-tl30*BET
                    table_data = {
                        "h5_cnt": h5_t, "h5_wr": h5_wr, "h5_pnl": round(sum(t.pnl for t in h5),1),
                        "h10_cnt": h10_t, "h10_wr": h10_wr, "h10_pnl": round(sum(t.pnl for t in h10),1),
                        "total_cnt": total_cnt, "total_wr": total_wr,
                        "total_pnl": round(sum(t.pnl for t in settled),1),
                        "today_cnt": today_t5+today_t10+today_t30,
                        "today_wr5": today_wr5, "today_wr10": today_wr10, "today_wr30": today_wr30,
                        "today_pnl": round(today_n5+today_n10+today_n30,1),
                        "today_detail_5": str(tw5)+"胜"+str(tl5)+"负",
                        "today_detail_10": str(tw10)+"胜"+str(tl10)+"负",
                        "today_detail_30": str(tw30)+"胜"+str(tl30)+"负",
                        "today_total_detail": str(tw5+tw10+tw30)+"胜"+str(tl5+tl10+tl30)+"负",
                        "trend": "上升" if latest.get("trend_up") else ("下降" if latest.get("trend_dn") else "震荡"),
                        "rsi_val": round(latest.get("rsi",0),1) if latest.get("rsi") else 0,
                        "bull5": latest.get("bull_5min",False), "bull15": latest.get("bull_15min",False),
                        "signal_text": "做多" if latest.get("buy_entry") else ("做空" if latest.get("sell_entry") else "等待"),
                    }
                    active_bets = [
                        {
                            "id": t.get("id", 0),
                            "time": t["time"].strftime("%H:%M:%S"),
                            "price": round(t["price"], 1),
                            "side": t["side"],
                            "timeframe": t["timeframe"],
                            "amount": t.get("amount", BET),
                            "remaining": max(0, int(TIMEFRAMES[t["timeframe"]] - (time.time() - t.get("placed_at", time.time())))),
                        }
                        for t in pending_trades if not t.get("settled")
                    ]
                    settled_bets = [
                        {"id": t.id, "time": t.time, "price": round(t.price, 1),
                         "side": t.side, "timeframe": t.timeframe,
                         "status": t.status, "pnl": round(t.pnl, 1)}
                        for t in trade_history[-50:] if t.is_manual
                    ][::-1]
                    data = json.dumps({
                        "latest": latest,
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "signal": "做多" if latest.get("buy_entry") else ("做空" if latest.get("sell_entry") else "等待"),
                        "trend": "上升" if latest.get("trend_up") else ("下降" if latest.get("trend_dn") else "震荡"),
                        "wr": round(wr, 1),
                        "total_pnl": round(total_pnl_val, 1),
                        "total": len(settled),
                        "cooldown_5": cooldown_5,
                        "cooldown_10": cooldown_10,
                        "table": table_data,
                        "ml": ml_result,
                        "active_bets": active_bets,
                        "settled_bets": settled_bets,
                        "capital": round(CAPITAL, 1),
                        "manual_markers": manual_markers[-100:],
                    })
                    yield f"data: {data}\n\n"
            time.sleep(1)
    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/bet", methods=["POST"])
def place_bet():
    """手动下单 API: POST {"side":"up"/"dn", "timeframe":"5min"/"10min"/"30min", "amount":10}"""
    global trade_id, pending_trades
    try:
        data = request.get_json()
        side = data.get("side")
        tf = data.get("timeframe")
        amount = float(data.get("amount", DEFAULT_BET))
        if side not in ("up", "dn") or tf not in TIMEFRAMES:
            return jsonify({"ok": False, "error": "参数错误"})
        if amount < 1:
            return jsonify({"ok": False, "error": "金额不能小于1U"})

        with cache["lock"]:
            df = cache["df"]
            if df is None:
                return jsonify({"ok": False, "error": "数据未就绪"})
            current_price = float(df.iloc[-1]["close"])
            current_bar = df.index[-1]
        now_str = current_bar.strftime("%H:%M:%S")

        with trade_lock:
            trade_id += 1
            trade_history.append(Trade(
                id=trade_id, time=now_str, price=current_price,
                side=side, timeframe=tf, status="pending", is_manual=True
            ))

        pending_trades.append({
            "id": trade_id, "time": current_bar, "price": current_price,
            "side": side, "timeframe": tf, "is_manual": True,
            "amount": amount, "tf_label": tf,
            "placed_at": time.time(),
        })

        # K线标记
        ts = int(current_bar.timestamp())
        manual_markers.append({"time": ts, "side": side, "price": round(current_price, 1), "amount": amount, "id": trade_id})

        direction = "涨" if side == "up" else "跌"
        print(f"[手动下单] #{trade_id} {tf}{direction} {amount}U @ {current_price:.0f}")
        return jsonify({
            "ok": True, "id": trade_id, "price": current_price,
            "side": side, "timeframe": tf, "amount": amount,
            "time_str": now_str, "capital": round(CAPITAL, 1),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/reset", methods=["POST"])
def reset_capital():
    """重置总资金到 1000U（二次确认由前端负责）"""
    global CAPITAL
    try:
        CAPITAL = 1000.0
        save_capital()
        # 清空今日统计
        global today_stats
        today_stats = {"5min_w": 0, "5min_l": 0, "10min_w": 0, "10min_l": 0, "30min_w": 0, "30min_l": 0}
        # 清空手动标记
        manual_markers.clear()
        print("[重置] 资金已重置为 1000U，统计已清零")
        return jsonify({"ok": True, "capital": 1000.0})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


HTML = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>BTC 策略监控</title>
    <script src="/static/lightweight-charts.standalone.production.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #1a1a2e; color: #ccc; font-family: 'Segoe UI', sans-serif; }
        .header { background: #16213e; padding: 12px 20px; display: flex; align-items: center; gap: 24px; border-bottom: 1px solid #333; }
        .header h1 { font-size: 18px; color: #f0b90b; }
        .header .info { display: flex; gap: 20px; font-size: 14px; }
        .header .info span { padding: 4px 10px; border-radius: 4px; background: #0f3460; }
        .signal-buy { color: #00ff88; font-weight: bold; }
        .signal-sell { color: #ff4444; font-weight: bold; }
        .signal-wait { color: #888; }
        .container { display: flex; height: calc(100vh - 50px); }
        .chart-area { flex: 1; display: flex; flex-direction: column; min-width: 0; }
        #chart { flex: 1; min-height: 200px; }
        .resize-handle { height: 4px; background: #333; cursor: ns-resize; flex-shrink: 0; }
        .resize-handle:hover { background: #f0b90b; }
        .rsi-pane { height: 120px; flex-shrink: 0; border-top: 1px solid #333; }
        .macd-pane { height: 80px; flex-shrink: 0; border-top: 1px solid #333; }
        .vol-pane { height: 70px; flex-shrink: 0; border-top: 1px solid #333; }
        .side-panel { width: 280px; background: #16213e; padding: 16px; overflow-y: auto; border-left: 1px solid #333; font-size: 13px; }
        .side-panel h3 { color: #f0b90b; font-size: 14px; margin-bottom: 8px; border-bottom: 1px solid #333; padding-bottom: 4px; }
        .stat-row { display: flex; justify-content: space-between; padding: 4px 0; }
        .stat-row .label { color: #888; }
        .stat-row .value { color: #fff; font-weight: 500; }
        .trade-log { margin-top: 12px; }
        .trade-item { padding: 6px 8px; margin: 4px 0; border-radius: 4px; font-size: 12px; border-left: 3px solid #555; }
        .trade-item.buy { border-left-color: #00ff88; background: rgba(0,255,136,0.08); }
        .trade-item.sell { border-left-color: #ff4444; background: rgba(255,68,68,0.08); }
        #time-display { color: #aaa; font-size: 12px; }
        .stats-table { position: absolute; bottom: 8px; right: 8px; font-size: 11px; border-collapse: collapse; background: rgba(26,26,46,0.85); border: 1px solid #444; pointer-events: none; z-index: 10; }
        .stats-table td { padding: 2px 6px; border: 1px solid #333; text-align: center; white-space: nowrap; }
        .stats-table .hl { color: #f0b90b; font-weight: bold; }
        .stats-table .gl { color: #888; }
        .stats-table .gn { color: #00ff88; }
        .stats-table .rd { color: #ff4444; }
        .tab-btn { padding: 4px 10px; border: 1px solid #333; border-radius: 4px; cursor: pointer; font-size: 11px; background: #1a1a2e; color: #888; flex:1; }
        .tab-btn.active { border-color: #f0b90b; color: #f0b90b; background: #f0b90b11; }
        .tab-btn:hover { border-color: #555; }
        .amt-btn { padding: 4px 6px; border: 1px solid #333; border-radius: 4px; cursor: pointer; font-size: 11px; background: #1a1a2e; color: #888; flex:1; text-align:center; }
        .amt-btn.active { border-color: #f0b90b; color: #f0b90b; background: #f0b90b11; }
        .amt-btn:hover { border-color: #555; }
        .btn-bet-up, .btn-bet-dn { flex:1; padding: 16px 0; border: none; border-radius: 8px; cursor: pointer; font-size: 18px; font-weight: bold; text-align:center; line-height:1.4; transition: opacity 0.2s; }
        .btn-bet-up { background: linear-gradient(180deg, #00e676 0%, #00c853 100%); color: #fff; }
        .btn-bet-up:hover { opacity: 0.85; }
        .btn-bet-dn { background: linear-gradient(180deg, #ff1744 0%, #d50000 100%); color: #fff; }
        .btn-bet-dn:hover { opacity: 0.85; }
        #bet-result { font-size: 12px; min-height: 20px; }
    </style>
</head>
<body>

<div class="header">
    <h1>BTC 策略监控</h1>
    <div class="info">
        <span id="disp-price">--</span>
        <span id="disp-signal" class="signal-wait">等待信号</span>
        <span id="disp-trend">--</span>
        <span id="disp-rsi">RSI: --</span>
        <span id="disp-wr">胜率: --</span>
        <span id="disp-pnl">PnL: --</span>
        <span id="disp-cooldown">冷却: --</span>
        <span id="time-display"></span>
        <label style="display:flex;align-items:center;gap:4px;font-size:12px;cursor:pointer;color:#888;margin-left:auto;">
            <input type="checkbox" id="ema-toggle" checked onchange="toggleEMA(this.checked)" style="accent-color:#f0b90b;"> EMA
        </label>
    </div>
</div>

<div class="container">
    <div class="chart-area">
        <div id="chart" style="position:relative;">
            <table class="stats-table" id="stats-table">
                <tr><td class="hl"></td><td class="hl">5m</td><td class="hl">10m</td><td class="hl">30m</td><td class="hl">合计</td><td class="hl">RSI</td><td class="hl">信号</td></tr>
                <tr><td class="gl">历史</td><td id="t-h5c">0</td><td id="t-h10c">0</td><td id="t-h30c">0</td><td id="t-tc">0</td><td id="t-rsi">--</td><td id="t-signal">等待</td></tr>
                <tr><td class="gl">胜率</td><td id="t-h5wr">0%</td><td id="t-h10wr">0%</td><td id="t-h30wr">0%</td><td id="t-twr">0%</td><td id="t-bull5">--</td><td id="t-bull15">--</td></tr>
                <tr><td class="gl">盈亏</td><td id="t-h5pnl">0U</td><td id="t-h10pnl">0U</td><td id="t-h30pnl">0U</td><td id="t-tpnl">0U</td><td colspan="2" id="t-status">暂无</td></tr>
                <tr><td class="hl">今日</td><td id="t-dcnt5">0单</td><td id="t-dcnt10">0单</td><td id="t-dcnt30">0单</td><td colspan="2" id="t-dpnl">0U</td><td id="t-dsignals">0次</td></tr>
                <tr><td class="gl">今日WR</td><td id="t-dwr5">0%</td><td id="t-dwr10">0%</td><td id="t-dwr30">0%</td><td id="td-total">0胜0负</td><td colspan="2" id="t-ddetail"></td></tr>
            </table>
        </div>
        <div class="resize-handle" id="resize-rsi"></div>
        <div class="rsi-pane" id="rsi-chart"></div>
        <div class="resize-handle" id="resize-macd"></div>
        <div class="macd-pane" id="macd-chart"></div>
        <div class="resize-handle" id="resize-vol"></div>
        <div class="vol-pane" id="vol-chart"></div>
    </div>
    <div class="side-panel" id="side-panel">
        <h3>实时状态</h3>
        <div class="stat-row"><span class="label">信号</span><span id="side-signal" class="signal-wait">等待</span></div>
        <div class="stat-row"><span class="label">趋势</span><span id="side-trend">--</span></div>
        <div class="stat-row"><span class="label">RSI(14)</span><span id="side-rsi">--</span></div>
        <div class="stat-row"><span class="label">RSI(6)</span><span id="side-rsi6">--</span></div>
        <div class="stat-row"><span class="label">价格</span><span id="side-price">--</span></div>
        <div class="stat-row"><span class="label">EMA8</span><span id="side-ema8">--</span></div>
        <div class="stat-row"><span class="label">EMA21</span><span id="side-ema21">--</span></div>
        <div class="stat-row"><span class="label">回踩深度</span><span id="side-bars-below">--</span></div>
        <div class="stat-row"><span class="label">反弹高度</span><span id="side-bars-above">--</span></div>
        <div class="stat-row"><span class="label">5分趋势</span><span id="side-bull5">--</span></div>
        <div class="stat-row"><span class="label">15分趋势</span><span id="side-bull15">--</span></div>
        <div class="stat-row"><span class="label">MACD柱</span><span id="side-macd">--</span></div>
        <div class="stat-row"><span class="label">ATR(14)</span><span id="side-atr">--</span></div>
        <div class="stat-row"><span class="label">放量</span><span id="side-vol">--</span></div>

        <h3 style="margin-top:16px;">手动预测</h3>
        <div id="bet-tabs" style="display:flex;gap:4px;margin-bottom:8px;">
            <button class="tab-btn active" onclick="switchTab('5min')">5分钟</button>
            <button class="tab-btn" onclick="switchTab('10min')">10分钟</button>
            <button class="tab-btn" onclick="switchTab('30min')">30分钟</button>
        </div>
        <div id="bet-price" style="text-align:center;font-size:20px;font-weight:bold;color:#fff;padding:4px 0;">--</div>
        <div style="display:flex;gap:4px;margin:6px 0;">
            <button class="amt-btn active" onclick="setAmt(10)">10U</button>
            <button class="amt-btn" onclick="setAmt(20)">20U</button>
            <button class="amt-btn" onclick="setAmt(50)">50U</button>
            <button class="amt-btn" onclick="setAmt(100)">100U</button>
        </div>
        <div style="display:flex;gap:8px;">
            <button class="btn-bet-up" id="btn-up" onclick="placeBet('up')">▲<br>涨</button>
            <button class="btn-bet-dn" id="btn-dn" onclick="placeBet('dn')">▼<br>跌</button>
        </div>
        <div id="bet-result" style="text-align:center;margin-top:4px;font-size:12px;"></div>

        <h3 style="margin-top:12px;">持仓</h3>
        <div id="active-bets" style="font-size:11px;max-height:150px;overflow-y:auto;"></div>

        <h3 style="margin-top:12px;">平仓历史</h3>
        <div id="settled-bets" style="font-size:11px;max-height:150px;overflow-y:auto;"></div>

        <div style="margin-top:12px;display:flex;justify-content:space-between;align-items:center;border-top:1px solid #333;padding-top:8px;">
            <div><span style="color:#888;font-size:11px;">总资金</span> <span id="disp-capital" style="color:#f0b90b;font-size:14px;font-weight:bold;">1000</span> <span style="color:#888;font-size:11px;">U</span></div>
            <button onclick="resetCapital()" style="padding:3px 10px;border:1px solid #ff444466;border-radius:4px;background:#ff444422;color:#ff4444;cursor:pointer;font-size:11px;">重置</button>
        </div>

        <h3 style="margin-top:16px;">ML 预测</h3>
        <div class="stat-row"><span class="label">5min</span><span id="ml-5">等待</span></div>
        <div class="stat-row"><span class="label">10min</span><span id="ml-10">等待</span></div>

        <h3 style="margin-top:16px;">交易统计</h3>
        <div class="stat-row"><span class="label">总交易</span><span id="side-total">--</span></div>
        <div class="stat-row"><span class="label">胜率</span><span id="side-wr">--</span></div>
        <div class="stat-row"><span class="label">总PnL</span><span id="side-pnl">--</span></div>
        <div class="stat-row"><span class="label">5min冷却</span><span id="side-cd5">--</span></div>
        <div class="stat-row"><span class="label">10min冷却</span><span id="side-cd10">--</span></div>

        <h3 style="margin-top:16px;">信号条件</h3>
        <div class="stat-row"><span class="label">pullback_buy</span><span id="cond-pb">✗</span></div>
        <div class="stat-row"><span class="label">range_buy</span><span id="cond-rb">✗</span></div>
        <div class="stat-row"><span class="label">buy_entry</span><span id="cond-be">✗</span></div>
        <div class="stat-row"><span class="label" style="padding-left:16px;">-> 5min涨</span><span id="cond-buy5">✗</span></div>
        <div class="stat-row"><span class="label" style="padding-left:16px;">-> 10min涨</span><span id="cond-buy10">✗</span></div>
        <div class="stat-row"><span class="label">bounce_sell</span><span id="cond-bs">✗</span></div>
        <div class="stat-row"><span class="label">range_sell</span><span id="cond-rs">✗</span></div>
        <div class="stat-row"><span class="label">sell_entry</span><span id="cond-se">✗</span></div>
        <div class="stat-row"><span class="label" style="padding-left:16px;">-> 5min跌</span><span id="cond-sell5">✗</span></div>
        <div class="stat-row"><span class="label" style="padding-left:16px;">-> 10min跌</span><span id="cond-sell10">✗</span></div>
        <div class="stat-row"><span class="label">bounce_sell</span><span id="cond-bs">✗</span></div>
        <div class="stat-row"><span class="label">range_buy</span><span id="cond-rb">✗</span></div>
        <div class="stat-row"><span class="label">range_sell</span><span id="cond-rs">✗</span></div>
        <div class="stat-row"><span class="label">buy_entry</span><span id="cond-be">✗</span></div>
        <div class="stat-row"><span class="label">sell_entry</span><span id="cond-se">✗</span></div>
    </div>
</div>

<script>
// ====== 初始化图表 ======
const chart = LightweightCharts.createChart(document.getElementById('chart'), {
    layout: { background: { color: '#1a1a2e' }, textColor: '#888' },
    grid: { vertLines: { color: '#222' }, horzLines: { color: '#222' } },
    timeScale: { timeVisible: true, secondsVisible: false },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderColor: '#333' },
});

const candleSeries = chart.addCandlestickSeries({
    upColor: '#00ff88', downColor: '#ff4444',
    borderDownColor: '#ff4444', borderUpColor: '#00ff88',
    wickDownColor: '#ff4444', wickUpColor: '#00ff88',
});

const ema8Series = chart.addLineSeries({ color: '#f0b90b', lineWidth: 2, title: 'EMA8' });
const ema21Series = chart.addLineSeries({ color: '#3b82f6', lineWidth: 1, title: 'EMA21' });
const ema50Series = chart.addLineSeries({ color: '#888', lineWidth: 1, title: 'EMA50' });

// RSI
const rsiChart = LightweightCharts.createChart(document.getElementById('rsi-chart'), {
    layout: { background: { color: '#1a1a2e' }, textColor: '#888' },
    grid: { vertLines: { color: '#222' }, horzLines: { color: '#222' } },
    timeScale: { visible: false },
    rightPriceScale: { borderColor: '#333' },
    height: 120,
});
const rsiSeries = rsiChart.addLineSeries({ color: '#a78bfa', lineWidth: 1.5, title: 'RSI(14)' });
const rsi6Series = rsiChart.addLineSeries({ color: '#f97316', lineWidth: 1, title: 'RSI(6)' });
// RSI 策略阈值线
const rsiBuyLine = rsiChart.addLineSeries({ color: '#00ff8844', lineWidth: 1, title: '做多上限50' });
const rsiSellLine = rsiChart.addLineSeries({ color: '#ff444444', lineWidth: 1, title: '做空下限45' });
const rsiOb = rsiChart.addLineSeries({ color: '#ff444422', lineWidth: 1 });
const rsiOs = rsiChart.addLineSeries({ color: '#00ff8822', lineWidth: 1 });

// MACD
const macdChart = LightweightCharts.createChart(document.getElementById('macd-chart'), {
    layout: { background: { color: '#1a1a2e' }, textColor: '#888' },
    grid: { vertLines: { color: '#222' }, horzLines: { color: '#222' } },
    timeScale: { visible: false },
    rightPriceScale: { borderColor: '#333' },
    height: 80,
});
const macdHistSeries = macdChart.addHistogramSeries({ title: 'MACD Hist' });
const macdLineSeries = macdChart.addLineSeries({ color: '#26a69a', lineWidth: 1.5, title: 'MACD' });
const macdSignalSeries = macdChart.addLineSeries({ color: '#ff6b6b', lineWidth: 1.5, title: 'SIGNAL' });

// Volume
const volChart = LightweightCharts.createChart(document.getElementById('vol-chart'), {
    layout: { background: { color: '#1a1a2e' }, textColor: '#888' },
    grid: { vertLines: { color: '#222' }, horzLines: { color: '#222' } },
    timeScale: { visible: false },
    rightPriceScale: { borderColor: '#333' },
    height: 70,
});
const volSeries = volChart.addHistogramSeries({ priceFormat: { type: 'volume' }, title: 'VOL' });
const volMa20Series = volChart.addLineSeries({ color: '#f0b90b66', lineWidth: 1, title: 'VOL MA20' });

// ====== 时间轴同步 (防循环) ======
let syncing = false;
const subPanes = [rsiChart, macdChart, volChart];
chart.timeScale().subscribeVisibleLogicalRangeChange(range => {
    if (range && !syncing) {
        syncing = true;
        subPanes.forEach(p => p.timeScale().setVisibleLogicalRange(range));
        syncing = false;
    }
});

// ====== 数据 ======
let allRecords = [];
let markers = [];

function updateChart(records) {
    if (!records || records.length === 0) return;

    const candles = records.map(r => ({ time: r.time, open: r.open, high: r.high, low: r.low, close: r.close }));
    candleSeries.setData(candles);

    const ema8data = records.filter(r => r.ema8 != null).map(r => ({ time: r.time, value: r.ema8 }));
    ema8Series.setData(ema8data);

    const ema21data = records.filter(r => r.ema21 != null).map(r => ({ time: r.time, value: r.ema21 }));
    ema21Series.setData(ema21data);

    const ema50data = records.filter(r => r.ema50 != null).map(r => ({ time: r.time, value: r.ema50 }));
    ema50Series.setData(ema50data);

    // RSI
    const rsidata = records.filter(r => r.rsi != null).map(r => ({ time: r.time, value: r.rsi }));
    rsiSeries.setData(rsidata);
    const rsi6data = records.filter(r => r.rsi6 != null).map(r => ({ time: r.time, value: r.rsi6 }));
    rsi6Series.setData(rsi6data);
    const rsiTimes = records.map(r => ({ time: r.time }));
    rsiBuyLine.setData(rsiTimes.map(t => ({ ...t, value: 50 })));
    rsiSellLine.setData(rsiTimes.map(t => ({ ...t, value: 45 })));
    rsiOb.setData(rsiTimes.map(t => ({ ...t, value: 70 })));
    rsiOs.setData(rsiTimes.map(t => ({ ...t, value: 30 })));

    // MACD
    const macdData = records.filter(r => r.macd_hist != null).map(r => ({
        time: r.time, value: r.macd_hist,
        color: r.macd_hist >= 0 ? '#26a69a66' : '#ef535066'
    }));
    macdHistSeries.setData(macdData);
    const macdLineData = records.filter(r => r.macd_line != null).map(r => ({ time: r.time, value: r.macd_line }));
    macdLineSeries.setData(macdLineData);
    const macdSigData = records.filter(r => r.macd_signal != null).map(r => ({ time: r.time, value: r.macd_signal }));
    macdSignalSeries.setData(macdSigData);

    // Volume
    const voldata = records.map(r => ({ time: r.time, value: r.volume, color: r.close >= r.open ? '#00ff8844' : '#ff444444' }));
    volSeries.setData(voldata);
    const voldata20 = records.filter(r => r.vol_ma20 != null).map(r => ({ time: r.time, value: r.vol_ma20 }));
    volMa20Series.setData(voldata20);

    // 信号标记
    const newMarkers = [];
    records.forEach(r => {
        if (r.buy5) {
            newMarkers.push({ time: r.time, position: 'belowBar', color: '#00ff88', shape: 'arrowUp', text: '5min涨' });
        }
        if (r.sell5) {
            newMarkers.push({ time: r.time, position: 'aboveBar', color: '#ff4444', shape: 'arrowDown', text: '5min跌' });
        }
        if (r.buy10) {
            newMarkers.push({ time: r.time, position: 'belowBar', color: '#2d7fff', shape: 'circle', text: '10min涨' });
        }
        if (r.sell10) {
            newMarkers.push({ time: r.time, position: 'aboveBar', color: '#cc00ff', shape: 'circle', text: '10min跌' });
        }
    });
    // 手动单标记 (红色箭头)
    if (window._manualMarkers) {
        window._manualMarkers.forEach(m => {
            newMarkers.push({
                time: m.time, position: m.side === 'up' ? 'belowBar' : 'aboveBar',
                color: m.side === 'up' ? '#ff6b6b' : '#ff6b6b',
                shape: m.side === 'up' ? 'arrowUp' : 'arrowDown',
                text: '手动' + (m.side === 'up' ? '涨' : '跌') + ' ' + m.amount + 'U',
            });
        });
    }
    candleSeries.setMarkers(newMarkers);
}

function updateSidePanel(latest) {
    if (!latest) return;

    const signal = latest.buy_entry ? '做多' : (latest.sell_entry ? '做空' : '等待');
    const trend = latest.trend_up ? '上升' : (latest.trend_dn ? '下降' : '震荡');

    document.getElementById('disp-price').textContent = '$' + latest.close.toFixed(0);
    document.getElementById('disp-signal').textContent = signal;
    document.getElementById('disp-signal').className = latest.buy_entry ? 'signal-buy' : (latest.sell_entry ? 'signal-sell' : 'signal-wait');
    document.getElementById('disp-trend').textContent = trend;
    document.getElementById('disp-trend').style.color = latest.trend_up ? '#00ff88' : (latest.trend_dn ? '#ff4444' : '#888');
    document.getElementById('disp-rsi').textContent = 'RSI: ' + (latest.rsi || '').toFixed(1);

    document.getElementById('side-signal').textContent = signal;
    document.getElementById('side-signal').className = latest.buy_entry ? 'signal-buy' : (latest.sell_entry ? 'signal-sell' : 'signal-wait');
    document.getElementById('side-trend').textContent = trend;
    document.getElementById('side-rsi').textContent = (latest.rsi || '').toFixed(1);
    document.getElementById('side-rsi6').textContent = (latest.rsi6 || '').toFixed(1);
    document.getElementById('side-price').textContent = latest.close.toFixed(0);
    document.getElementById('bet-price').textContent = '$' + latest.close.toFixed(0);
    document.getElementById('side-ema8').textContent = (latest.ema8 || '').toFixed(0);
    document.getElementById('side-ema21').textContent = (latest.ema21 || '').toFixed(0);
    document.getElementById('side-bars-below').textContent = latest.bars_below + ' / 3';
    document.getElementById('side-bars-above').textContent = latest.bars_above + ' / 3';
    document.getElementById('side-bull5').textContent = latest.bull_5min ? '📈涨' : '📉跌';
    document.getElementById('side-bull15').textContent = latest.bull_15min ? '📈涨' : '📉跌';
    document.getElementById('side-macd').textContent = latest.macd_hist > 0 ? '📊正' : '📊负';
    document.getElementById('side-atr').textContent = latest.atr14 ? latest.atr14.toFixed(0) : '--';

    const volOk = latest.volume > (latest.vol_ma20 || 0);
    document.getElementById('side-vol').textContent = volOk ? '✅' : '❌';

    // 信号条件
    document.getElementById('cond-pb').textContent = latest.pullback_buy ? '✅' : '✗';
    document.getElementById('cond-pb').style.color = latest.pullback_buy ? '#00ff88' : '#666';
    document.getElementById('cond-rb').textContent = latest.range_buy ? '✅' : '✗';
    document.getElementById('cond-rb').style.color = latest.range_buy ? '#00ff88' : '#666';
    document.getElementById('cond-be').textContent = latest.buy_entry ? '✅' : '✗';
    document.getElementById('cond-be').style.color = latest.buy_entry ? '#00ff88' : '#666';
    document.getElementById('cond-buy5').textContent = latest.buy5 ? '✅' : '✗';
    document.getElementById('cond-buy5').style.color = latest.buy5 ? '#00ff88' : '#666';
    document.getElementById('cond-buy10').textContent = latest.buy10 ? '✅' : '✗';
    document.getElementById('cond-buy10').style.color = latest.buy10 ? '#2d7fff' : '#666';
    document.getElementById('cond-bs').textContent = latest.bounce_sell ? '✅' : '✗';
    document.getElementById('cond-bs').style.color = latest.bounce_sell ? '#ff4444' : '#666';
    document.getElementById('cond-rs').textContent = latest.range_sell ? '✅' : '✗';
    document.getElementById('cond-rs').style.color = latest.range_sell ? '#ff4444' : '#666';
    document.getElementById('cond-se').textContent = latest.sell_entry ? '✅' : '✗';
    document.getElementById('cond-se').style.color = latest.sell_entry ? '#ff4444' : '#666';
    document.getElementById('cond-sell5').textContent = latest.sell5 ? '✅' : '✗';
    document.getElementById('cond-sell5').style.color = latest.sell5 ? '#ff4444' : '#666';
    document.getElementById('cond-sell10').textContent = latest.sell10 ? '✅' : '✗';
    document.getElementById('cond-sell10').style.color = latest.sell10 ? '#cc00ff' : '#666';
}

function updateTradeStats(stats) {
    if (!stats) return;
    document.getElementById('side-total').textContent = stats.total || 0;
    document.getElementById('side-wr').textContent = (stats.win_rate || 0) + '%';
    const pnl = stats.total_pnl || 0;
    document.getElementById('side-pnl').textContent = pnl.toFixed(1) + 'U';
    document.getElementById('side-pnl').style.color = pnl >= 0 ? '#00ff88' : '#ff4444';
}

function updateCooldown(cd5, cd10) {
    document.getElementById('side-cd5').textContent = cd5 + ' bars';
    document.getElementById('side-cd10').textContent = cd10 + ' bars';
    document.getElementById('disp-cooldown').textContent = cd5 + '/' + cd10;
}

// ====== 手动预测 (币安风格) ======
let currentTf = '5min';
let currentAmt = 10;

function switchTab(tf) {
    currentTf = tf;
    document.querySelectorAll('#bet-tabs .tab-btn').forEach(btn => {
        btn.classList.toggle('active', btn.textContent.includes(tf.replace('min','')));
    });
}
function setAmt(amt) {
    currentAmt = amt;
    document.querySelectorAll('.amt-btn').forEach(btn => {
        btn.classList.toggle('active', parseInt(btn.textContent) === amt);
    });
}
function placeBet(side) {
    const el = document.getElementById('bet-result');
    el.textContent = '下单中...';
    el.style.color = '#888';
    fetch('/api/bet', { method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({side, timeframe: currentTf, amount: currentAmt})
    })
    .then(r => r.json())
    .then(data => {
        if (data.ok) {
            const dir = side === 'up' ? '涨' : '跌';
            el.textContent = '✅ #' + data.id + ' ' + data.timeframe + dir + ' @' + data.price.toFixed(0);
            el.style.color = '#00ff88';
        } else {
            el.textContent = '❌ ' + data.error;
            el.style.color = '#ff4444';
        }
    })
    .catch(err => { el.textContent = '❌ ' + err; el.style.color = '#ff4444'; });
}

function updateBets(active, settled) {
    // 持仓
    const ab = document.getElementById('active-bets');
    if (!active || active.length === 0) {
        ab.innerHTML = '<div style="color:#666;padding:4px 0;">暂无持仓</div>';
    } else {
        let html = '<table style="width:100%;border-collapse:collapse;"><tr style="color:#888;font-size:10px;"><td>时间</td><td>方向</td><td>开仓价</td><td>金额</td><td>剩余</td></tr>';
        active.forEach(t => {
            const dir = t.side === 'up' ? '<span style="color:#00ff88;">▲涨</span>' : '<span style="color:#ff4444;">▼跌</span>';
            const m = Math.floor(t.remaining / 60);
            const s = t.remaining % 60;
            html += '<tr><td>' + t.time + '</td><td>' + dir + '</td><td>' + t.price + '</td><td>' + t.amount + 'U</td><td>' + m + ':' + (s<10?'0':'') + s + '</td></tr>';
        });
        html += '</table>';
        ab.innerHTML = html;
    }

    // 平仓历史
    const sb = document.getElementById('settled-bets');
    if (!settled || settled.length === 0) {
        sb.innerHTML = '<div style="color:#666;padding:4px 0;">暂无记录</div>';
    } else {
        let html = '<table style="width:100%;border-collapse:collapse;"><tr style="color:#888;font-size:10px;"><td>时间</td><td>周期</td><td>方向</td><td>开仓</td><td>结果</td><td>PnL</td></tr>';
        settled.slice(0, 20).forEach(t => {
            const dir = t.side === 'up' ? '<span style="color:#00ff88;">涨</span>' : '<span style="color:#ff4444;">跌</span>';
            const res = t.status === 'won' ? '<span style="color:#00ff88;">✅胜</span>' : '<span style="color:#ff4444;">❌负</span>';
            const pnlColor = t.pnl >= 0 ? '#00ff88' : '#ff4444';
            html += '<tr><td>' + t.time + '</td><td>' + t.timeframe + '</td><td>' + dir + '</td><td>' + t.price + '</td><td>' + res + '</td><td style="color:' + pnlColor + ';">' + (t.pnl > 0 ? '+' : '') + t.pnl + 'U</td></tr>';
        });
        html += '</table>';
        sb.innerHTML = html;
    }
}

function updateML(ml) {
    if (!ml) return;
    const el5 = document.getElementById('ml-5');
    const el10 = document.getElementById('ml-10');
    if (ml.pred5 === 1) { el5.textContent = '涨 (' + ml.conf5 + '%)'; el5.style.color = '#00ff88'; }
    else if (ml.pred5 === 0) { el5.textContent = '跌 (' + ml.conf5 + '%)'; el5.style.color = '#ff4444'; }
    else { el5.textContent = '等待'; el5.style.color = '#888'; }
    if (ml.pred10 === 1) { el10.textContent = '涨 (' + ml.conf10 + '%)'; el10.style.color = '#00ff88'; }
    else if (ml.pred10 === 0) { el10.textContent = '跌 (' + ml.conf10 + '%)'; el10.style.color = '#ff4444'; }
    else { el10.textContent = '等待'; el10.style.color = '#888'; }
}

function updateStatsTable(t) {
    if (!t) return;
    const s = v => v ?? '--';
    document.getElementById('t-h5c').textContent = s(t.h5_cnt);
    document.getElementById('t-h10c').textContent = s(t.h10_cnt);
    document.getElementById('t-h30c').textContent = '0';
    document.getElementById('t-tc').textContent = s(t.total_cnt);
    document.getElementById('t-h5wr').textContent = s(t.h5_wr) + '%';
    document.getElementById('t-h10wr').textContent = s(t.h10_wr) + '%';
    document.getElementById('t-h30wr').textContent = '--';
    document.getElementById('t-twr').textContent = s(t.total_wr) + '%';
    document.getElementById('t-h5pnl').textContent = s(t.h5_pnl) + 'U';
    document.getElementById('t-h10pnl').textContent = s(t.h10_pnl) + 'U';
    document.getElementById('t-h30pnl').textContent = '0U';
    document.getElementById('t-tpnl').textContent = s(t.total_pnl) + 'U';
    document.getElementById('t-rsi').textContent = s(t.rsi_val);
    document.getElementById('t-signal').textContent = s(t.signal_text);
    document.getElementById('t-signal').style.color = t.signal_text === '做多' ? '#00ff88' : (t.signal_text === '做空' ? '#ff4444' : '#888');
    document.getElementById('t-bull5').textContent = t.bull5 ? '5分涨' : '5分跌';
    document.getElementById('t-bull5').style.color = t.bull5 ? '#00ff88' : '#ff4444';
    document.getElementById('t-bull15').textContent = t.bull15 ? '15分涨' : '15分跌';
    document.getElementById('t-bull15').style.color = t.bull15 ? '#00ff88' : '#ff4444';
    const pnl = t.total_pnl || 0;
    document.getElementById('t-status').textContent = pnl >= 0 && t.total_cnt > 0 ? '✅盈利' : (t.total_cnt > 0 ? '❌亏损' : '暂无');
    document.getElementById('t-status').colSpan = '2';
    document.getElementById('t-dcnt5').textContent = s(t.today_detail_5);
    document.getElementById('t-dcnt10').textContent = s(t.today_detail_10);
    document.getElementById('t-dcnt30').textContent = s(t.today_detail_30);
    document.getElementById('t-dwr5').textContent = s(t.today_wr5) + '%';
    document.getElementById('t-dwr10').textContent = s(t.today_wr10) + '%';
    document.getElementById('t-dwr30').textContent = s(t.today_wr30) + '%';
    document.getElementById('t-dpnl').textContent = s(t.today_pnl) + 'U';
    document.getElementById('t-dpnl').style.color = (t.today_pnl || 0) >= 0 ? '#00ff88' : '#ff4444';
    document.getElementById('td-total').textContent = s(t.today_total_detail);
    document.getElementById('t-ddetail').textContent = '';
}

function updateTradeHistory(trades) {
    const panel = document.getElementById('side-panel');
    // Remove old trade log
    const oldLog = document.getElementById('trade-log');
    if (oldLog) oldLog.remove();
    if (!trades || trades.length === 0) return;
    const div = document.createElement('div');
    div.id = 'trade-log';
    div.className = 'trade-log';
    const h3 = document.createElement('h3');
    h3.style.cssText = 'color:#f0b90b;font-size:14px;margin:12px 0 8px;border-bottom:1px solid #333;padding-bottom:4px;';
    h3.textContent = '交易记录';
    div.appendChild(h3);
    trades.slice(0, 30).forEach(t => {
        const item = document.createElement('div');
        item.className = 'trade-item ' + (t.side === 'up' ? 'buy' : 'sell');
        const status = t.status === 'pending' ? '⏳' : (t.status === 'won' ? '✅' : '❌');
        const direction = t.side === 'up' ? '做多' : '做空';
        item.textContent = '[' + t.timeframe + '] ' + t.time + ' ' + direction + ' @' + t.price.toFixed(0) + ' ' + status;
        if (t.status !== 'pending') {
            item.textContent += ' ' + (t.pnl > 0 ? '+' : '') + t.pnl.toFixed(0) + 'U';
        }
        div.appendChild(item);
    });
    panel.appendChild(div);
}

// ====== 初始化加载 ======
fetch('/api/data')
    .then(r => r.json())
    .then(data => {
        allRecords = data.records;
        updateChart(data.records);
        updateSidePanel(data.latest);
        updateTradeStats(data.stats);
        updateTradeHistory(data.trades);
        updateCooldown(data.cooldown_5, data.cooldown_10);
        updateStatsTable(data.table);
        updateML(data.ml);
        if (data.active_bets || data.settled_bets) updateBets(data.active_bets, data.settled_bets);
        if (data.capital) document.getElementById('disp-capital').textContent = data.capital;
        if (data.manual_markers) window._manualMarkers = data.manual_markers;
    });

// ====== SSE 实时推送 ======
const evtSource = new EventSource('/api/stream');
evtSource.onmessage = (e) => {
    try {
        const data = JSON.parse(e.data);
        updateSidePanel(data.latest);
        document.getElementById('disp-wr').textContent = '胜率: ' + (data.wr || 0) + '%';
        document.getElementById('disp-pnl').textContent = 'PnL: ' + (data.total_pnl || 0).toFixed(1) + 'U';
        document.getElementById('disp-pnl').style.color = (data.total_pnl || 0) >= 0 ? '#00ff88' : '#ff4444';
        document.getElementById('time-display').textContent = '更新: ' + data.time;
        updateCooldown(data.cooldown_5 || 0, data.cooldown_10 || 0);
        if (data.table) updateStatsTable(data.table);
        if (data.ml) updateML(data.ml);
        if (data.active_bets || data.settled_bets) updateBets(data.active_bets, data.settled_bets);
        if (data.capital) document.getElementById('disp-capital').textContent = data.capital;
        if (data.manual_markers) window._manualMarkers = data.manual_markers;
    } catch (err) {}
};

// ====== 定时刷新完整数据 (每30秒) ======
setInterval(() => {
    fetch('/api/data')
        .then(r => r.json())
        .then(data => {
            if (data.records && data.records.length > 0) {
                allRecords = data.records;
                updateChart(data.records);
                updateSidePanel(data.latest);
                updateTradeStats(data.stats);
                updateTradeHistory(data.trades);
                updateStatsTable(data.table);
                updateML(data.ml);
                if (data.active_bets || data.settled_bets) updateBets(data.active_bets, data.settled_bets);
                if (data.capital) document.getElementById('disp-capital').textContent = data.capital;
                if (data.manual_markers) window._manualMarkers = data.manual_markers;
            }
        });
}, 30000);

// ====== EMA 开关 ======
function toggleEMA(show) {
    [ema8Series, ema21Series, ema50Series].forEach(s => s.applyOptions({ visible: show }));
}

// ====== 资金重置 ======
function resetCapital() {
    if (!confirm('确定要重置为 1000U 吗？所有手动标记和今日统计将被清空。')) return;
    fetch('/api/reset', { method: 'POST' })
        .then(r => r.json())
        .then(d => {
            if (d.ok) {
                document.getElementById('disp-capital').textContent = '1000';
                window._manualMarkers = [];
                alert('✅ 已重置为 1000U');
            } else alert('❌ ' + d.error);
        });
}

// ====== 面板拖拽调整大小 ======
function makeResizable(handleId, chartInstance) {
    const handle = document.getElementById(handleId);
    let startY, startH;

    handle.addEventListener('mousedown', (e) => {
        const pane = handle.nextElementSibling; // 拖拽手柄下面的面板
        if (!pane) return;
        startY = e.clientY;
        startH = pane.offsetHeight;
        document.body.style.cursor = 'ns-resize';
        document.body.style.userSelect = 'none';

        const onMove = (ev) => {
            const dh = ev.clientY - startY;
            const newH = Math.max(50, startH + dh);
            pane.style.height = newH + 'px';
            if (chartInstance) chartInstance.applyOptions({ height: newH });
        };
        const onUp = () => {
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            document.removeEventListener('mousemove', onMove);
            document.removeEventListener('mouseup', onUp);
        };
        document.addEventListener('mousemove', onMove);
        document.addEventListener('mouseup', onUp);
    });
}

makeResizable('resize-rsi', rsiChart);
makeResizable('resize-macd', macdChart);
makeResizable('resize-vol', volChart);

// ====== 窗口自适应 ======
function resizeCharts() {
    const w = document.querySelector('.chart-area').clientWidth;
    const chartEl = document.getElementById('chart');
    chart.applyOptions({ width: w, height: chartEl.offsetHeight });
    rsiChart.applyOptions({ width: w });
    macdChart.applyOptions({ width: w });
    volChart.applyOptions({ width: w });
}
window.addEventListener('resize', resizeCharts);
setTimeout(resizeCharts, 100);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    print("[Web] BTC 监控面板启动: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
