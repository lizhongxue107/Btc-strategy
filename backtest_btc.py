"""
BTC 5min/10min 分钟级预测回测 + 自动参数优化
=============================================
基于投票评分制, 每分钟用多项指标综合打分,
预测未来 5/10 分钟涨跌. 网格搜索最优参数使胜率 > 60%.

用法:
  python backtest_btc.py run              # 下载+回测+优化
  python backtest_btc.py optimize          # 用已有数据只跑优化
  python backtest_btc.py fetch             # 只下载数据
  python backtest_btc.py run --days 60    # 指定天数
"""

import os, sys, json, time
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

DATA_FILE = "btc_1m_data.pkl"
PARAMS_FILE = "best_params.json"


# ==============================
# 1. 数据下载
# ==============================

def fetch_binance_klines(symbol="BTCUSDT", interval="1m", days=30):
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_k = []
    cur = start_ts
    hosts = ["api.binance.com", "api1.binance.com", "api2.binance.com", "api3.binance.com"]
    while cur < end_ts:
        ok = False
        for host in hosts:
            try:
                r = requests.get(
                    f"https://{host}/api/v3/klines",
                    params={"symbol": symbol, "interval": interval,
                            "startTime": cur, "limit": 1000},
                    timeout=10, verify=False,
                )
                r.raise_for_status()
                data = r.json()
                if data:
                    all_k.extend(data)
                cur = data[-1][0] + 1 if data else cur + 60000
                ok = True
                break
            except Exception:
                continue
        if not ok:
            raise RuntimeError("all binance endpoints failed")
        print(f"  fetched {len(all_k)} klines, cur={datetime.fromtimestamp(cur/1000)}")
    df = pd.DataFrame(all_k, columns=[
        "ts", "open", "high", "low", "close", "vol",
        "close_time", "qvol", "trades", "taker_buy_vol", "taker_buy_qvol", "ignore",
    ])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms")
    for c in ["open", "high", "low", "close", "vol"]:
        df[c] = df[c].astype(float)
    df.set_index("ts", inplace=True)
    return df


MODEL_FILE = "ml_model.pkl"


def _fetch_futures_series(path, params, retries=3):
    """获取期货数据序列, 自动重试."""
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


def _fetch_chunked(path, params_template, ts_field, val_field, val_type, start_ts, end_ts, chunk_days=5):
    """
    分块获取期货数据, 避免单次请求跨度太大导致 API 拒绝.
    """
    all_rows = []
    chunk_ms = chunk_days * 86400000
    cur = start_ts
    while cur < end_ts:
        chunk_end = min(cur + chunk_ms, end_ts)
        params = dict(params_template)
        params["startTime"] = cur
        params["endTime"] = chunk_end
        raw = _fetch_futures_series(path, params)
        if not raw:
            cur = chunk_end + 1
            continue
        all_rows.extend(raw)
        if len(raw) < 500:
            cur = chunk_end + 1
        else:
            cur = raw[-1][ts_field] + 1
    return all_rows


def fetch_futures_data(symbol="BTCUSDT", days=30, kline_index=None):
    """
    下载期货数据: 资金费率, OI, Taker 比, 多空比.
    返回 DataFrame 对齐到 kline_index (含所有 1-min 时间戳).
    """
    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    if kline_index is not None:
        idx = kline_index
    else:
        idx = pd.date_range(start=pd.Timestamp(start_ts, unit="ms"),
                           end=pd.Timestamp(end_ts, unit="ms"), freq="1min")

    all_series = {}

    # 1. Funding rate (每 8 小时)
    print("  fetching funding rate...")
    fr = _fetch_chunked("/fapi/v1/fundingRate",
        {"symbol": symbol, "limit": 1000}, "fundingTime", "fundingRate", float,
        start_ts, end_ts, chunk_days=30)
    if fr:
        s = pd.DataFrame(fr).set_index(
            pd.to_datetime([r["fundingTime"] for r in fr], unit="ms")
        )["fundingRate"].astype(float)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s.reindex(idx, method="ffill").bfill()
        all_series["funding_rate"] = s
        print(f"    {len(fr)} records")
    else:
        print("    empty")

    # 2. Open Interest History (5min)
    print("  fetching open interest history...")
    oi = _fetch_chunked("/futures/data/openInterestHist",
        {"symbol": symbol, "period": "5m", "limit": 500}, "timestamp", "sumOpenInterest", float,
        start_ts, end_ts, chunk_days=5)
    if oi:
        s = pd.DataFrame(oi).set_index(
            pd.to_datetime([r["timestamp"] for r in oi], unit="ms")
        )["sumOpenInterest"].astype(float)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s.reindex(idx, method="ffill").bfill()
        all_series["open_interest"] = s
        all_series["oi_change"] = s.pct_change().fillna(0)
        print(f"    {len(oi)} records")
    else:
        print("    empty")

    # 3. Taker Buy/Sell Ratio (5min)
    print("  fetching taker buy/sell ratio...")
    taker = _fetch_chunked("/futures/data/takerlongshortRatio",
        {"symbol": symbol, "period": "5m", "limit": 500}, "timestamp", "buySellRatio", float,
        start_ts, end_ts, chunk_days=5)
    if taker:
        s = pd.DataFrame(taker).set_index(
            pd.to_datetime([r["timestamp"] for r in taker], unit="ms")
        )["buySellRatio"].astype(float)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s.reindex(idx, method="ffill").bfill()
        all_series["taker_buy_ratio"] = s
        print(f"    {len(taker)} records")
    else:
        print("    empty")

    # 4. Long/Short Account Ratio (5min)
    print("  fetching long/short ratio...")
    ls = _fetch_chunked("/futures/data/globalLongShortAccountRatio",
        {"symbol": symbol, "period": "5m", "limit": 500}, "timestamp", "longShortRatio", float,
        start_ts, end_ts, chunk_days=5)
    if ls:
        s = pd.DataFrame(ls).set_index(
            pd.to_datetime([r["timestamp"] for r in ls], unit="ms")
        )["longShortRatio"].astype(float)
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s = s.reindex(idx, method="ffill").bfill()
        all_series["ls_ratio"] = s
        print(f"    {len(ls)} records")
    else:
        print("    empty")

    if not all_series:
        print("  WARNING: no futures data fetched")
        return pd.DataFrame(index=idx)

    result = pd.DataFrame(all_series, index=idx)
    print(f"  futures data: {len(result)} rows, {len(result.columns)} cols: {list(result.columns)}")
    return result


def merge_futures_features(df, futures_df):
    """将期货特征合并到 K 线 DataFrame 中."""
    if futures_df.empty:
        return df
    for col in futures_df.columns:
        df[col] = futures_df[col]
    return df


# ==============================
# 2. 指标计算
# ==============================

def emma(arr, period):
    """numpy EMA"""
    alpha = 2.0 / (period + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def compute_indicators(df):
    """计算所有指标, 返回添加列的 DataFrame"""
    c = df["close"].values
    v = df["vol"].values
    n = len(c)

    # EMA
    df["ema8"] = emma(c, 8)
    df["ema21"] = emma(c, 21)
    df["ema50"] = emma(c, 50)
    df["above_ema8"] = c > df["ema8"].values

    # RSI14
    diff = np.diff(c)
    gain = np.where(diff > 0, diff, 0)
    loss = np.where(diff < 0, -diff, 0)
    avg_gain = np.empty(n)
    avg_loss = np.empty(n)
    avg_gain[0] = 0
    avg_loss[0] = 0
    for i in range(1, n):
        avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i - 1]) / 14
        avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i - 1]) / 14
    rs = np.where(avg_loss == 0, 100, np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0))
    df["rsi14"] = 100 - 100 / (1 + rs)
    df["rsi14_rising"] = np.concatenate([[False], np.diff(df["rsi14"].values) > 0])

    # RSI6
    avg_gain6 = np.empty(n)
    avg_loss6 = np.empty(n)
    avg_gain6[0] = 0
    avg_loss6[0] = 0
    for i in range(1, n):
        avg_gain6[i] = (avg_gain6[i - 1] * 5 + gain[i - 1]) / 6
        avg_loss6[i] = (avg_loss6[i - 1] * 5 + loss[i - 1]) / 6
    rs6 = np.where(avg_loss6 == 0, 100, np.divide(avg_gain6, avg_loss6, out=np.zeros_like(avg_gain6), where=avg_loss6 != 0))
    df["rsi6"] = 100 - 100 / (1 + rs6)
    df["rsi6_rising"] = np.concatenate([[False], np.diff(df["rsi6"].values) > 0])

    # MACD
    ema12 = emma(c, 12)
    ema26 = emma(c, 26)
    macd_line = ema12 - ema26
    signal = emma(macd_line, 9)
    df["macd"] = macd_line
    df["macd_signal"] = signal
    df["macd_hist"] = macd_line - signal
    df["macd_above_signal"] = macd_line > signal

    # Volume SMA20
    vol_sma20 = np.empty(n)
    for i in range(n):
        vol_sma20[i] = np.mean(v[max(0, i - 19):i + 1])
    df["vol_sma20"] = vol_sma20
    df["vol_ratio"] = v / np.where(vol_sma20 == 0, 1, vol_sma20)
    df["high_vol"] = df["vol_ratio"].values > 1.2

    # ATR
    high = df["high"].values
    low = df["low"].values
    prev_close = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = np.empty(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = (atr[i - 1] * 13 + tr[i]) / 14
    df["atr"] = atr

    # --- 短时动量特征 ---
    # 过去 3/5 分钟价格变化
    df["mom_3"] = np.concatenate([[0, 0], c[2:] - c[:-2]])
    df["mom_5"] = np.concatenate([[0] * 4, c[4:] - c[:-4]])

    # 连续收阳/收阴 K 线计数
    o = df["open"].values
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

    # K 线实体比例 (close-open)/(high-low), [-1, +1]
    hl = high - low
    hl = np.where(hl == 0, 1, hl)  # 避免除零
    df["candle_body_ratio"] = (c - o) / hl

    # 最近 5 根 K 线的高低点及价格位置
    high_5 = np.zeros(n)
    low_5 = np.zeros(n)
    for i in range(n):
        high_5[i] = np.max(high[max(0, i - 4):i + 1])
        low_5[i] = np.min(low[max(0, i - 4):i + 1])
    df["high_5"] = high_5
    df["low_5"] = low_5
    hl5 = high_5 - low_5
    hl5 = np.where(hl5 == 0, 1, hl5)
    df["price_position_5"] = (c - low_5) / hl5

    return df


# ==============================
# 3. 投票评分系统
# ==============================

def compute_net_score(df, vol_min=1.2, bars_threshold=3):
    """
    计算每分钟的净评分 (net_score).
    新的 8 条件投票系统:
    1. price_position_5: 趋势位置
    2. mom_3: 短时动量方向
    3. 连续同向 K 线
    4. candle_body_ratio + vol: 强动量
    5. 区间突破 + vol
    6. RSI6 极端
    7. 成交量确认
    8. MACD 辅助
    总分范围 [-10, +10]
    """
    c = df["close"].values
    n = len(c)

    net = np.zeros(n, dtype=np.int8)

    for i in range(n):
        score = 0

        # --- 1. 价格位置 (趋势位置) ---
        pp5 = df["price_position_5"].iloc[i]
        if pp5 > 0.7:
            score += 1
        elif pp5 < 0.3:
            score -= 1

        # --- 2. 3 分钟动量方向 ---
        if df["mom_3"].iloc[i] > 0:
            score += 1
        elif df["mom_3"].iloc[i] < 0:
            score -= 1

        # --- 3. 连续同向 K 线 ---
        if df["bull_bars"].iloc[i] >= 3:
            score += 1
        elif df["bear_bars"].iloc[i] >= 3:
            score -= 1

        # --- 4. 强动量 + 成交量确认 ---
        cbr = df["candle_body_ratio"].iloc[i]
        vol_ratio = df["vol_ratio"].iloc[i]
        if cbr > 0.5 and vol_ratio > vol_min:
            score += 2
        elif cbr < -0.5 and vol_ratio > vol_min:
            score -= 2

        # --- 5. 区间突破 ---
        if i > 0:
            prev_high = df["high_5"].iloc[i - 1]
            prev_low = df["low_5"].iloc[i - 1]
            if df["close"].iloc[i] > prev_high and vol_ratio > 1.5:
                score += 2
            elif df["close"].iloc[i] < prev_low and vol_ratio > 1.5:
                score -= 2

        # --- 6. RSI6 极端 ---
        r6 = df["rsi6"].iloc[i]
        if r6 < 25 and df["rsi6_rising"].iloc[i]:
            score += 1
        elif r6 > 75 and not df["rsi6_rising"].iloc[i]:
            score -= 1

        # --- 7. 成交量确认 ---
        if vol_ratio > vol_min:
            if df["above_ema8"].iloc[i]:
                score += 1
            else:
                score -= 1

        # --- 8. MACD 辅助 ---
        if df["macd_hist"].iloc[i] > 0 and df["macd_hist"].iloc[i] > df["macd_hist"].iloc[max(i - 1, 0)]:
            score += 1
        elif df["macd_hist"].iloc[i] < 0 and df["macd_hist"].iloc[i] < df["macd_hist"].iloc[max(i - 1, 0)]:
            score -= 1

        net[i] = np.clip(score, -10, 10)

    return net


# ==============================
# 4. 快速评估 (纯 numpy)
# ==============================

def _fast_eval(net_score, close, threshold, cool5, cool10):
    """快速评估一组参数, 返回 (w5, l5, w10, l10)"""
    n = len(net_score)
    pred_5 = np.full(n, -1, dtype=np.int8)
    pred_10 = np.full(n, -1, dtype=np.int8)
    c5, c10 = 0, 0
    for i in range(n):
        if c5 > 0:
            c5 -= 1
        if c10 > 0:
            c10 -= 1
        score = net_score[i]
        if c5 == 0:
            if score >= threshold:
                pred_5[i] = 1
                c5 = cool5
            elif score <= -threshold:
                pred_5[i] = 0
                c5 = cool5
        if c10 == 0:
            if score >= threshold:
                pred_10[i] = 1
                c10 = cool10
            elif score <= -threshold:
                pred_10[i] = 0
                c10 = cool10

    hit_5 = np.zeros(n, dtype=bool)
    hit_10 = np.zeros(n, dtype=bool)
    for i in range(n - 5):
        if pred_5[i] != -1:
            hit_5[i] = (pred_5[i] == 1 and close[i + 5] > close[i]) or (pred_5[i] == 0 and close[i + 5] < close[i])
    for i in range(n - 10):
        if pred_10[i] != -1:
            hit_10[i] = (pred_10[i] == 1 and close[i + 10] > close[i]) or (pred_10[i] == 0 and close[i + 10] < close[i])

    w5 = int(hit_5.sum())
    t5 = int((pred_5 != -1).sum())
    w10 = int(hit_10.sum())
    t10 = int((pred_10 != -1).sum())
    return w5, t5 - w5, w10, t10 - w10


# ==============================
# 5. 回测主函数
# ==============================

def run_backtest(df, threshold=3, vol_min=1.2, bars_threshold=3,
                 cool5=5, cool10=10, min_signals=30):
    """
    运行回测, 返回 dict 包含胜率等指标.
    """
    net_score = compute_net_score(df, vol_min, bars_threshold)
    close = df["close"].values
    w5, l5, w10, l10 = _fast_eval(net_score, close, threshold, cool5, cool10)
    t5 = w5 + l5
    t10 = w10 + l10

    if t5 < min_signals:
        return {"win5": 0, "loss5": 0, "total5": 0, "rate5": 0,
                "win10": 0, "loss10": 0, "total10": 0, "rate10": 0,
                "params": {"threshold": threshold, "vol_min": vol_min,
                          "bars_threshold": bars_threshold, "cool5": cool5, "cool10": cool10}}

    return {
        "win5": w5, "loss5": l5, "total5": t5, "rate5": round(w5 / t5 * 100, 1),
        "win10": w10, "loss10": l10, "total10": t10, "rate10": round(w10 / t10 * 100, 1),
        "params": {"threshold": threshold, "vol_min": vol_min,
                  "bars_threshold": bars_threshold, "cool5": cool5, "cool10": cool10},
    }


# ==============================
# 6. 参数优化 (网格搜索)
# ==============================

def optimize(df, min_signals=30, verbose=True):
    """
    网格搜索最优参数.
    使用 net_cache 预计算 net_score, 避免重复计算.
    """
    param_grid = {
        "threshold": [2, 3, 4, 5],
        "vol_min": [1.0, 1.2, 1.5],
        "bars_threshold": [1, 2, 3],
        "cool5": [3, 5, 8],
        "cool10": [5, 10, 15],
    }

    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))
    total = len(combos)
    print(f"total combos: {total}")
    print()

    # 预计算: net_score 只依赖 vol_min + bars_threshold
    # threshold/cool5/cool10 不影响 net_score
    close = df["close"].values
    net_cache = {}
    vol_mins = sorted(set(v for _, v, _, _, _ in combos))
    bar_ths = sorted(set(b for _, _, b, _, _ in combos))
    for vm in vol_mins:
        for bt in bar_ths:
            net_cache[(vm, bt)] = compute_net_score(df, vm, bt)

    results = []
    for idx, (th, vm, bt, c5, c10) in enumerate(combos):
        net_score = net_cache[(vm, bt)]
        w5, l5, w10, l10 = _fast_eval(net_score, close, th, c5, c10)
        t5 = w5 + l5
        t10 = w10 + l10

        if t5 >= min_signals or t10 >= min_signals:
            results.append({
                "win5": w5, "loss5": l5, "total5": t5,
                "rate5": round(w5 / t5 * 100, 1) if t5 else 0,
                "win10": w10, "loss10": l10, "total10": t10,
                "rate10": round(w10 / t10 * 100, 1) if t10 else 0,
                "params": {"threshold": th, "vol_min": vm,
                          "bars_threshold": bt, "cool5": c5, "cool10": c10},
            })

        if verbose and (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{total} evaluated, {len(results)} qualified")

    if not results:
        print("no qualified results (min_signals too high?)")
        return []

    # 按综合得分排序 (5min 胜率 * 信号数 + 10min 胜率 * 信号数)
    for r in results:
        r["score"] = r["rate5"] * r["total5"] + r["rate10"] * r["total10"]
    results.sort(key=lambda x: x["score"], reverse=True)

    return results


# ==============================
# 7. 打印结果
# ==============================

def print_results(results, top_n=20):
    """打印优化结果排行榜"""
    print(f"\n{'=' * 100}")
    print(f"{'Rank':<5} {'thr':<4} {'vol':<5} {'bar':<5} {'c5':<5} {'c10':<5}"
          f" {'W5':<5} {'L5':<5} {'T5':<6} {'R5%':<6}"
          f" {'W10':<5} {'L10':<5} {'T10':<6} {'R10%':<6} {'Score':<8}")
    print(f"{'-' * 100}")
    for i, r in enumerate(results[:top_n]):
        p = r["params"]
        print(f"{i + 1:<5} {p['threshold']:<4} {p['vol_min']:<5} {p['bars_threshold']:<5}"
              f" {p['cool5']:<5} {p['cool10']:<5}"
              f" {r['win5']:<5} {r['loss5']:<5} {r['total5']:<6} {r['rate5']:<6}"
              f" {r['win10']:<5} {r['loss10']:<5} {r['total10']:<6} {r['rate10']:<6}"
              f" {r['score']:<8.0f}")
    print(f"{'=' * 100}")

    # 找出 5min 胜率 > 60% 的最佳参数
    good5 = [r for r in results if r["rate5"] >= 60 and r["total5"] >= 30]
    if good5:
        print(f"\n>>> {len(good5)} combos with 5min win rate >= 60% <<<")
        best5 = good5[0]
        print(f"Best 5min: threshold={best5['params']['threshold']}, "
              f"vol_min={best5['params']['vol_min']}, "
              f"bars_threshold={best5['params']['bars_threshold']}, "
              f"cool5={best5['params']['cool5']}, cool10={best5['params']['cool10']}, "
              f"win={best5['rate5']}% ({best5['win5']}/{best5['total5']})")

    good10 = [r for r in results if r["rate10"] >= 60 and r["total10"] >= 30]
    if good10:
        print(f"\n>>> {len(good10)} combos with 10min win rate >= 60% <<<")
        best10 = good10[0]
        print(f"Best 10min: threshold={best10['params']['threshold']}, "
              f"vol_min={best10['params']['vol_min']}, "
              f"bars_threshold={best10['params']['bars_threshold']}, "
              f"cool5={best10['params']['cool5']}, cool10={best10['params']['cool10']}, "
              f"win={best10['rate5']}% ({best10['win5']}/{best10['total5']})")


def save_best(results, filepath=PARAMS_FILE):
    """保存最优参数到 JSON"""
    if not results:
        return
    good5 = [r for r in results if r["rate5"] >= 60 and r["total5"] >= 30]
    if not good5:
        # 胜率不足 60%, 保存综合分最高的
        with open(filepath, "w") as f:
            json.dump(results[0], f, indent=2, ensure_ascii=False)
        print(f"no combo with >60% win rate, saved best overall to {filepath}")
        return

    best = good5[0]
    with open(filepath, "w") as f:
        json.dump(best, f, indent=2, ensure_ascii=False)
    print(f"saved best params to {filepath}")


# ==============================
# 8. 机器学习预测 (RandomForest)
# ==============================

def prepare_ml_data(df, pred_horizon=5, train_ratio=0.6):
    """
    准备 ML 训练/测试数据.
    特征: 所有数值型指标列.
    目标: close[i+pred_horizon] > close[i] (1) or not (0).
    按时间顺序分割, 前 train_ratio 训练, 后 1-train_ratio 测试.
    """
    close = df["close"].values
    n = len(df)

    # 选择特征列 (排除原始价格/成交量列和 meta 列)
    exclude = {"ts", "open", "high", "low", "close", "vol",
               "close_time", "qvol", "trades", "taker_buy_vol",
               "taker_buy_qvol", "ignore", "high_vol"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]
    print(f"ML feature columns ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].values.astype(np.float64)
    y = np.zeros(n, dtype=np.int8)
    for i in range(n - pred_horizon):
        y[i] = 1 if close[i + pred_horizon] > close[i] else 0

    # 去掉尾部 pred_horizon 行 (没有目标值)
    X = X[:n - pred_horizon]
    y = y[:n - pred_horizon]
    n_usable = len(X)

    # 去掉包含 NaN 的行
    mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    # 时间序列分割
    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"ML data: total={len(X)}, train={len(X_train)}, test={len(X_test)}")
    print(f"  y5 baseline (always 1): train={y_train.mean():.1%}, test={y_test.mean():.1%}")

    return X_train, X_test, y_train, y_test, feature_cols


def ml_optimize(df, pred_horizon=5, train_ratio=0.6):
    """
    使用 RandomForest 训练预测模型.
    评估含概率阈值过滤: 只有 RF 置信度 > threshold 才计为预测.
    检查不同波动率区间的准确率差异.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test, features = prepare_ml_data(df, pred_horizon, train_ratio)
    close_test = df["close"].values
    atr_test = df["atr"].values

    # 训练模型
    model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_t = time.time() - t0

    # 测试集预测概率
    proba = model.predict_proba(X_test)
    pos_proba = proba[:, 1]  # P(up)
    y_pred = model.predict(X_test)
    base_acc = accuracy_score(y_test, y_pred)
    print(f"\nRF (n=200, depth=6): train={train_t:.1f}s")
    print(f"  baseline accuracy (50% threshold): {base_acc:.1%}")

    # ---- 概率阈值分析 ----
    print(f"\n  Probability threshold analysis:")
    print(f"  {'Thresh':<8} {'Acc':<8} {'Signals':<10} {'% of test':<10}")
    print(f"  {'-' * 36}")
    for th in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        preds = (pos_proba >= th).astype(int)
        # 只评估有信号的样本
        mask = (pos_proba >= th) | (pos_proba <= 1 - th)
        n_sig = mask.sum()
        if n_sig < 10:
            continue
        acc = accuracy_score(y_test[mask], preds[mask])
        print(f"  {th:<8} {acc:<8.1%} {n_sig:<10} {n_sig / len(y_test):<10.1%}")

    # ---- 波动率分段评估 ----
    atr_median = np.median(atr_test)
    # 映射测试集索引
    test_start = int(len(X_train) * 1.0)  # after NaN removal in prepare_ml_data
    # Actually, the indexes don't map cleanly. Let me re-compute.
    # We need alignment between X_test and the source df.
    # prepare_ml_data removes NaN rows, so indexes are shifted.
    # For a simpler approach, use the net_score buy/sell regime idea.

    # ---- 特征重要性 ----
    imp = model.feature_importances_
    top_idx = np.argsort(imp)[::-1][:10]
    print(f"\n  Top 10 features:")
    for idx in top_idx:
        print(f"    {features[idx]}: {imp[idx]:.4f}")

    return model, features


def ml_evaluate_regime(df, model, features, pred_horizon=5, train_ratio=0.6):
    """
    只在测试集上按波动率高低分段评估.
    """
    from sklearn.metrics import accuracy_score

    close = df["close"].values
    atr = df["atr"].values
    n = len(df)

    X = df[features].values.astype(np.float64)
    y = np.zeros(n, dtype=np.int8)
    for i in range(n - pred_horizon):
        y[i] = 1 if close[i + pred_horizon] > close[i] else 0
    X = X[:n - pred_horizon]
    y = y[:n - pred_horizon]
    mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    # 时间分割: test 在后 40%
    split = int(len(X) * train_ratio)
    X_test = X[split:]
    y_test = y[split:]
    atr_test = atr[len(atr) - len(y):][split:]  # 对齐

    proba = model.predict_proba(X_test)
    pos_proba = proba[:, 1]

    atr_med = np.median(atr_test)

    print(f"\n  Test-set regime analysis (split={train_ratio:.0%}/{1-train_ratio:.0%}):")
    regimes = [
        ("all", slice(None)),
        ("ATR > median", atr_test > atr_med),
        ("ATR < median", atr_test <= atr_med),
        ("ATR > 1.2*median", atr_test > 1.2 * atr_med),
        ("ATR > 1.5*median", atr_test > 1.5 * atr_med),
        ("ATR > 2.0*median", atr_test > 2.0 * atr_med),
    ]
    print(f"  {'Regime':<18} {'Acc@55%':<10} {'Signals':<10} {'%test':<8} {'Baseline':<10}")
    print(f"  {'-' * 56}")
    for name, cond in regimes:
        idx = np.where(cond)[0] if name != "all" else np.arange(len(y_test))
        if len(idx) < 10:
            continue
        pred = (pos_proba[idx] >= 0.55).astype(int)
        acc = accuracy_score(y_test[idx], pred)
        n_sig = len(idx)
        print(f"  {name:<18} {acc:<10.1%} {n_sig:<10} {n_sig/len(y_test):<8.1%} {y_test[idx].mean():<10.1%}")


def ml_predict_live(df, model, pred_horizon=5):
    """
    用已训练的模型对最新数据做预测.
    返回 (prediction, proba) 或 None.
    """
    exclude = {"ts", "open", "high", "low", "close", "vol",
               "close_time", "qvol", "trades", "taker_buy_vol",
               "taker_buy_qvol", "ignore", "high_vol"}
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]
    last = df[feature_cols].iloc[-1:].values.astype(np.float64)
    if np.isnan(last).any():
        return None, None
    pred = model.predict(last)[0]
    proba = model.predict_proba(last)[0][1] if hasattr(model, "predict_proba") else None
    return pred, proba


# ==============================
# 9. CLI
# ==============================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="BTC 5min/10min 回测优化")
    parser.add_argument("action", nargs="?", default="run", choices=["run", "optimize", "fetch", "ml", "train"])
    parser.add_argument("--days", type=int, default=30, help="回测天数")
    parser.add_argument("--min-signals", type=int, default=30, help="最少信号数")
    parser.add_argument("--top-n", type=int, default=20, help="显示前 N 个结果")
    args = parser.parse_args()

    if args.action == "fetch":
        print(f"downloading {args.days} days of 1m data...")
        df = fetch_binance_klines("BTCUSDT", "1m", args.days)
        df = compute_indicators(df)
        df.to_pickle(DATA_FILE)
        print(f"saved {len(df)} rows to {DATA_FILE}")
        return

    if args.action == "optimize" and not os.path.exists(DATA_FILE):
        print(f"{DATA_FILE} not found, run 'python backtest_btc.py fetch' first")
        sys.exit(1)

    if args.action == "ml":
        if not os.path.exists(DATA_FILE):
            print(f"{DATA_FILE} not found, run 'python backtest_btc.py fetch' first")
            sys.exit(1)
        df = pd.read_pickle(DATA_FILE)
        print(f"data shape: {df.shape}, period: {df.index[0]} ~ {df.index[-1]}")

        # 获取期货数据 (如果还没有)
        if "funding_rate" not in df.columns:
            print("fetching futures data...")
            futures_df = fetch_futures_data("BTCUSDT", args.days, kline_index=df.index)
            df = merge_futures_features(df, futures_df)
            print(f"enhanced data shape: {df.shape}, cols={list(df.columns)}")

        print(f"\n--- 5min prediction ---")
        t0 = time.time()
        model50, features50 = ml_optimize(df, pred_horizon=5, train_ratio=0.6)
        ml_evaluate_regime(df, model50, features50, pred_horizon=5, train_ratio=0.6)
        t5 = time.time() - t0
        print(f"\n--- 10min prediction ---")
        model100, features100 = ml_optimize(df, pred_horizon=10, train_ratio=0.6)
        ml_evaluate_regime(df, model100, features100, pred_horizon=10, train_ratio=0.6)
        t10 = time.time() - t0
        print(f"\ntotal time: {t10:.1f}s")
        return

    if args.action == "train":
        import pickle
        from sklearn.ensemble import RandomForestClassifier

        if not os.path.exists(DATA_FILE):
            print(f"downloading data first...")
            df = fetch_binance_klines("BTCUSDT", "1m", args.days)
            df = compute_indicators(df)
            df.to_pickle(DATA_FILE)
        else:
            df = pd.read_pickle(DATA_FILE)
        print(f"data: {df.shape}, {df.index[0]} ~ {df.index[-1]}")

        if "funding_rate" not in df.columns:
            print("fetching futures data...")
            futures_df = fetch_futures_data("BTCUSDT", args.days, kline_index=df.index)
            df = merge_futures_features(df, futures_df)

        # 特征列
        exclude = {"ts", "open", "high", "low", "close", "vol",
                   "close_time", "qvol", "trades", "taker_buy_vol",
                   "taker_buy_qvol", "ignore", "high_vol"}
        feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in (np.float64, np.int64, np.float32, np.int32)]
        print(f"features ({len(feature_cols)}): {feature_cols}")

        close = df["close"].values
        X = df[feature_cols].values.astype(np.float64)
        n = len(X)
        y5 = np.zeros(n)
        y10 = np.zeros(n)
        for i in range(n - 5):
            y5[i] = 1 if close[i + 5] > close[i] else 0
        for i in range(n - 10):
            y10[i] = 1 if close[i + 10] > close[i] else 0
        mask = ~np.isnan(X).any(axis=1)
        X, y5, y10 = X[mask], y5[mask], y10[mask]
        split = int(len(X) * 0.6)
        print(f"samples: {len(X)}, train: {split}, test: {len(X)-split}")

        model_5 = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
        model_10 = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)

        print("training 5min model...")
        model_5.fit(X[:split], y5[:split])
        acc5 = model_5.score(X[split:], y5[split:])
        print(f"  5min test accuracy: {acc5:.3f} ({acc5:.1%})")

        print("training 10min model...")
        model_10.fit(X[:split], y10[:split])
        acc10 = model_10.score(X[split:], y10[split:])
        print(f"  10min test accuracy: {acc10:.3f} ({acc10:.1%})")

        # Retrain on all
        print("retraining on full data...")
        model_5.fit(X, y5)
        model_10.fit(X, y10)

        payload = {
            "model_5": model_5,
            "model_10": model_10,
            "features": feature_cols,
            "acc5": acc5,
            "acc10": acc10,
            "train_date": datetime.now().isoformat(),
        }
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(payload, f)
        print(f"saved model to {MODEL_FILE}")
        return

    if args.action in ("run", "optimize"):
        if args.action == "run":
            print(f"downloading {args.days} days of 1m data...")
            df = fetch_binance_klines("BTCUSDT", "1m", args.days)
            df = compute_indicators(df)
            df.to_pickle(DATA_FILE)
            print(f"saved {len(df)} rows to {DATA_FILE}")
        else:
            print(f"loading {DATA_FILE}...")
            df = pd.read_pickle(DATA_FILE)

        print(f"data shape: {df.shape}, period: {df.index[0]} ~ {df.index[-1]}")
        print(f"running grid search optimization...")
        print(f"  threshold: [2, 3, 4]")
        print(f"  vol_min: [1.0, 1.2, 1.5]")
        print(f"  bars_threshold: [1, 2, 3]")
        print(f"  cool5: [3, 5, 8]")
        print(f"  cool10: [5, 10, 15]")
        t0 = time.time()
        results = optimize(df, args.min_signals)
        elapsed = time.time() - t0
        print(f"optimization took {elapsed:.1f}s")

        if results:
            print_results(results, args.top_n)
            save_best(results)
        else:
            print("no qualified results found")
        return


if __name__ == "__main__":
    main()
