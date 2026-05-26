# BTC 5min/10min 策略回测脚本
# 下载币安历史数据并模拟策略

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta

# ========== 1. 下载数据 ==========
def fetch_binance_klines(symbol="BTCUSDT", interval="1m", days=30):
    """从币安下载历史K线数据"""
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_klines = []
    current_start = start_time

    while current_start < end_time:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": 1000
        }
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            klines = resp.json()
            if not klines:
                break
            all_klines.extend(klines)
            current_start = klines[-1][0] + 1  # 下一批次的开头
            time.sleep(0.1)  # 避免触发频率限制
        except Exception as e:
            print(f"获取数据出错: {e}")
            break

    df = pd.DataFrame(all_klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df.set_index("timestamp", inplace=True)
    print(f"下载完成: {len(df)} 根 {interval}K线 ({days}天)")
    return df

# ========== 2. 计算指标 ==========
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

def bb(close, period=20, std_dev=2.0):
    basis = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = basis + std * std_dev
    lower = basis - std * std_dev
    return basis, upper, lower

def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def sma(series, period):
    return series.rolling(period).mean()

# ========== 3. 策略逻辑 ==========
def compute_indicators(df):
    """计算所有公共指标"""
    df["ema8"] = ema(df["close"], 8)
    df["ema21"] = ema(df["close"], 21)
    df["ema50"] = ema(df["close"], 50)
    df["rsi14"] = rsi(df["close"], 14)
    df["rsi6"] = rsi(df["close"], 6)
    _, _, df["macd_hist"] = macd(df["close"])
    df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
    df["vol_ma20"] = sma(df["volume"], 20)

    basis, bb_u, bb_d = bb(df["close"])
    df["bb_u"] = bb_u
    df["bb_d"] = bb_d
    df["bb_width"] = bb_u - bb_d
    df["bbw_sma"] = sma(df["bb_width"], 20)

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

    # ATR中位数过滤器 (避免横盘)
    df["atr_above_median"] = df["atr14"] > df["atr14"].rolling(100).median()

    # MACD柱上升/下降
    df["macd_rising"] = df["macd_hist"] > df["macd_hist"].shift(1)
    df["macd_falling"] = df["macd_hist"] < df["macd_hist"].shift(1)

    # 趋势方向和回踩深度
    df["trend_up"] = (df["ema8"] > df["ema21"]) & (df["ema8"] > df["ema8"].shift(1))
    df["trend_dn"] = (df["ema8"] < df["ema21"]) & (df["ema8"] < df["ema8"].shift(1))

    lookback = 3
    for i in range(1, lookback + 1):
        df[f"close_below_{i}"] = (df["close"].shift(i) < df["ema8"].shift(i)).astype(int)
        df[f"close_above_{i}"] = (df["close"].shift(i) > df["ema8"].shift(i)).astype(int)
    df["bars_below"] = sum(df[f"close_below_{i}"] for i in range(1, lookback + 1))
    df["bars_above"] = sum(df[f"close_above_{i}"] for i in range(1, lookback + 1))

    return df


def run_backtest(df, params=None):
    """执行策略回测, params控制参数"""
    if params is None:
        params = {
            "bars_threshold": 2,
            "rsi_buy_max": 65,
            "rsi_sell_min": 35,
            "vol_min": 1.0,
            "atr_filter": False,
            "macd_filter": False,
            "cool5": 10,
            "cool10": 20,
        }
    bt = params

    if not bt.get("quiet", False):
        print(f"\n回测数据: {df.index[0]} -> {df.index[-1]}")

    # 信号
    df["pullback_buy"] = (
        df["trend_up"] &
        (df["bars_below"] >= bt["bars_threshold"]) &
        (df["close"] > df["ema8"]) &
        (df["close"] > df["open"]) &
        (df["volume"] > df["vol_ma20"] * bt["vol_min"]) &
        (df["rsi14"] > df["rsi14"].shift(1)) &
        (df["rsi14"] < bt["rsi_buy_max"]) &
        ((df["atr_above_median"]) if bt["atr_filter"] else True) &
        ((df["macd_rising"]) if bt["macd_filter"] else True)
    )

    df["bounce_sell"] = (
        df["trend_dn"] &
        (df["bars_above"] >= bt["bars_threshold"]) &
        (df["close"] < df["ema8"]) &
        (df["close"] < df["open"]) &
        (df["volume"] > df["vol_ma20"] * bt["vol_min"]) &
        (df["rsi14"] < df["rsi14"].shift(1)) &
        (df["rsi14"] > bt["rsi_sell_min"]) &
        ((df["atr_above_median"]) if bt["atr_filter"] else True) &
        ((df["macd_falling"]) if bt["macd_filter"] else True)
    )

    # 多时间框架过滤 + 冷却
    df["buy5_raw"] = df["pullback_buy"] & df["bull_5min"]
    df["sell5_raw"] = df["bounce_sell"] & ~df["bull_5min"]
    df["buy10_raw"] = df["pullback_buy"] & df["bull_5min"] & df["bull_15min"]
    df["sell10_raw"] = df["bounce_sell"] & ~df["bull_5min"] & ~df["bull_15min"]

    # 冷却系统
    cool5 = bt["cool5"]
    cool10 = bt["cool10"]
    df["buy5"] = False
    df["sell5"] = False
    df["buy10"] = False
    df["sell10"] = False

    c5 = 0
    c10 = 0
    for i in range(len(df)):
        if c5 > 0:
            c5 -= 1
        if c10 > 0:
            c10 -= 1

        if c5 == 0:
            if df["buy5_raw"].iloc[i]:
                df.loc[df.index[i], "buy5"] = True
                c5 = cool5
            elif df["sell5_raw"].iloc[i]:
                df.loc[df.index[i], "sell5"] = True
                c5 = cool5

        if c10 == 0:
            if df["buy10_raw"].iloc[i]:
                df.loc[df.index[i], "buy10"] = True
                c10 = cool10
            elif df["sell10_raw"].iloc[i]:
                df.loc[df.index[i], "sell10"] = True
                c10 = cool10

    # ========== 结算 ==========
    # 记录方向
    df["d5"] = np.where(df["buy5"], "up", np.where(df["sell5"], "dn", None))
    df["d10"] = np.where(df["buy10"], "up", np.where(df["sell10"], "dn", None))

    w5, l5 = 0, 0
    for idx in df[df["buy5"]].index:
        pos = df.index.get_loc(idx)
        if pos + 5 < len(df):
            if df.iloc[pos + 5]["close"] > df.loc[idx, "close"]:
                w5 += 1
            else:
                l5 += 1
    for idx in df[df["sell5"]].index:
        pos = df.index.get_loc(idx)
        if pos + 5 < len(df):
            if df.iloc[pos + 5]["close"] < df.loc[idx, "close"]:
                w5 += 1
            else:
                l5 += 1

    w10, l10 = 0, 0
    for idx in df[df["buy10"]].index:
        pos = df.index.get_loc(idx)
        if pos + 10 < len(df):
            if df.iloc[pos + 10]["close"] > df.loc[idx, "close"]:
                w10 += 1
            else:
                l10 += 1
    for idx in df[df["sell10"]].index:
        pos = df.index.get_loc(idx)
        if pos + 10 < len(df):
            if df.iloc[pos + 10]["close"] < df.loc[idx, "close"]:
                w10 += 1
            else:
                l10 += 1

    return df, (w5, l5), (w10, l10)

# ========== 4. 打印结果 ==========
def print_results(df, res5, res10):
    w5, l5 = res5
    w10, l10 = res10
    t5 = w5 + l5
    t10 = w10 + l10

    bet = 50
    pr_rate = 0.8
    n5 = w5 * (bet * pr_rate) - l5 * bet
    n10 = w10 * (bet * pr_rate) - l10 * bet
    total_pnl = n5 + n10
    wr5 = w5 / t5 * 100 if t5 > 0 else 0
    wr10 = w10 / t10 * 100 if t10 > 0 else 0
    total_wr = (w5 + w10) / (t5 + t10) * 100 if (t5 + t10) > 0 else 0

    print("\n" + "=" * 50)
    print("        BTC 策略回测结果 (v6.1)")
    print(f"        数据范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
    print("=" * 50)
    print(f"{'':15} {'5min':>8} {'10min':>8} {'合计':>8}")
    print("-" * 42)
    print(f"{'次数':15} {t5:>8} {t10:>8} {t5+t10:>8}")
    print(f"{'胜率':15} {wr5:>7.1f}% {wr10:>7.1f}% {total_wr:>7.1f}%")
    print(f"{'净盈亏(U)':15} {n5:>8.1f} {n10:>8.1f} {total_pnl:>8.1f}")
    print("-" * 42)

    # 信号频率
    total_hours = (df.index[-1] - df.index[0]).total_seconds() / 3600
    print(f"\n信号频率: 5min={t5/total_hours:.1f}/天 ({t5/len(df)*1440*60:.1f}/小时)")
    print(f"总交易小时: {total_hours:.0f}h")

    # 买/卖分别统计
    buy_count = df["buy5"].sum() + df["buy10"].sum()
    sell_count = df["sell5"].sum() + df["sell10"].sum()
    print(f"\n做多信号: {int(buy_count)} | 做空信号: {int(sell_count)}")

    # 阈值尝试
    print("\n--- 参数敏感性分析 ---")
    for name, wr in [("5分钟", wr5), ("10分钟", wr10), ("合计", total_wr)]:
        status = "达标!" if wr >= 60 else "不达标"
        print(f"{name}胜率: {wr:.1f}% {status}")

# ========== 5. 主函数 ==========
def main():
    print("正在下载BTC 1分钟历史数据...")
    df = fetch_binance_klines("BTCUSDT", "1m", days=30)

    if len(df) < 5000:
        print(f"数据太少 ({len(df)}), 尝试下载更长时间...")
        df = fetch_binance_klines("BTCUSDT", "1m", days=60)

    print(f"正在计算指标 ({len(df)} 根K线)...")
    df = compute_indicators(df)
    df, res5, res10 = run_backtest(df)
    print_results(df, res5, res10)

    # 参数优化扫描
    print("\n" + "=" * 60)
    print("参数优化扫描中...")
    print("=" * 60)
    optimize(df)

def optimize(df):
    """网格搜索最优参数"""
    param_grid = {
        "bars_threshold": [2, 3, 4],
        "rsi_buy_max": [50, 55, 60, 65],
        "rsi_sell_min": [30, 35, 40, 45],
        "vol_min": [1.0, 1.2, 1.5],
        "atr_filter": [False, True],
        "macd_filter": [False, True],
    }

    keys = list(param_grid.keys())
    results = []

    # 笛卡尔积
    from itertools import product
    combinations = list(product(*[param_grid[k] for k in keys]))
    total = len(combinations)
    print(f"共计 {total} 种参数组合...")

    for idx, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        params["cool5"] = 10
        params["cool10"] = 20
        params["quiet"] = True
        # 合理性检查: rsi_buy_max must be > rsi_sell_min
        if params["rsi_buy_max"] <= params["rsi_sell_min"]:
            continue
        # atr+macd都需要预计算完毕
        df_copy = df.copy()
        _, (w5, l5), (w10, l10) = run_backtest(df_copy, params)
        t5 = w5 + l5
        t10 = w10 + l10
        if t5 + t10 < 20:  # 信号太少没统计意义
            continue
        wr5 = w5 / t5 * 100 if t5 > 0 else 0
        wr10 = w10 / t10 * 100 if t10 > 0 else 0
        total_wr = (w5 + w10) / (t5 + t10) * 100
        results.append((total_wr, wr5, wr10, t5, t10, params))
        if (idx + 1) % 50 == 0 and len(results) > 0:
            best = max(results, key=lambda r: r[0])
            print(f"  [{idx+1}/{total}] 当前最佳: {best[0]:.1f}% ({best[5]})")

    # 排序取前20
    results.sort(key=lambda r: r[0], reverse=True)
    print(f"\n{'='*60}")
    print(f"参数优化结果 TOP 20")
    print(f"{'='*60}")
    print(f"{'排名':<4} {'合计胜率':<8} {'5min':<7} {'10min':<7} {'5次':<5} {'10次':<5} 参数")
    print(f"{'-'*60}")
    for i, (tw, w5, w10, t5, t10, p) in enumerate(results[:20]):
        print(f"{i+1:<4} {tw:<7.1f}% {w5:<6.1f}% {w10:<6.1f}% {t5:<5} {t10:<5} {p}")

    # 最佳参数写回Pine Script
    best_params = results[0][5]
    print(f"\n最佳参数: {best_params}")
    print("\n更新 Pine Script 策略参数...")

if __name__ == "__main__":
    main()
