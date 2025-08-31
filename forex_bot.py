#!/usr/bin/env python3
"""
mt5_ai_multi_tf_longshort.py
Multi-timeframe (M1, M5, M15) AI forex bot for MetaTrader5 (long & short).
- Trains/loads three calibrated GradientBoosting models (one per TF)
- Ensemble decision: average probability + per-TF votes
- Symmetric long & short logic
- ATR-based SL/TP and position sizing
- Daily loss cap, one active position (long or short) at a time
USAGE: run while MT5 terminal is open and logged in.
"""

import as mt5
import pandas as pd, numpy as np, time, math, os
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from joblib import dump, load

# ---------------- CONFIG ----------------
SYMBOL = "EURUSD"
TF_MAP = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15}
HISTORY_BARS = {"M1": 15000, "M5": 4000, "M15": 2000}
MODEL_PATHS = {"M1":"model_m1.joblib", "M5":"model_m5.joblib", "M15":"model_m15.joblib"}
TRAIN_SPLITS = 4

# Ensemble thresholds (symmetric for long/short)
ENTRY_THRESHOLD = 0.58   # average proba to enter LONG (proba = chance of upward move)
TF_THRESH = 0.55         # per-TF proba threshold for voting (for long)
ENTRY_THRESHOLD_SHORT = 1.0 - ENTRY_THRESHOLD  # symmetric for short entries
TF_THRESH_SHORT = 1.0 - TF_THRESH
EXIT_THRESHOLD = 0.50

RISK_PER_TRADE = 0.003   # 0.3% equity per trade
SL_ATR_MULT = 2.5
TP_ATR_MULT = 3.0
SLIPPAGE = 10
DAILY_LOSS_CAP = 0.02    # 2% equity
POLL_SECONDS = 3
MAGIC = 20250829
# ----------------------------------------

def log(*a):
    print(datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)

def mt5_init():
    if not mt5.initialize():
        raise RuntimeError("mt5.initialize() failed: " + str(mt5.last_error()))
    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"symbol_select({SYMBOL}) failed")

def fetch_rates(tf_key, n):
    tf = TF_MAP[tf_key]
    rates = mt5.copy_rates_from_pos(SYMBOL, tf, 0, n)
    if rates is None:
        raise RuntimeError(f"copy_rates failed for {tf_key}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.rename(columns={'open':'Open','high':'High','low':'Low','close':'Close','tick_volume':'Volume'}, inplace=True)
    return df[['Open','High','Low','Close','Volume']]

def make_features(df):
    X = pd.DataFrame(index=df.index)
    X['ret_1'] = df['Close'].pct_change(1)
    X['ret_4'] = df['Close'].pct_change(4)
    X['ret_16'] = df['Close'].pct_change(16)
    X['sma50'] = df['Close'].rolling(50).mean()
    X['sma200'] = df['Close'].rolling(200).mean()
    X['dist_sma50'] = df['Close']/X['sma50'] - 1
    X['dist_sma200'] = df['Close']/X['sma200'] - 1
    X['rsi'] = RSIIndicator(df['Close'], 14).rsi()
    X['atr'] = AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range()
    X['vol_48'] = df['Close'].pct_change().rolling(48).std()
    X['hour'] = df.index.tz_convert('UTC').hour
    X['dow']  = df.index.tz_convert('UTC').dayofweek
    X = X.dropna()
    # target: 4-bar ahead direction (binary up/down)
    y = (df['Close'].shift(-4)/df['Close'] - 1).reindex(X.index)
    y = (y > 0).astype(int)
    return X, y

def train_model_for_tf(tf_key):
    log("Training model for", tf_key)
    df = fetch_rates(tf_key, HISTORY_BARS[tf_key])
    X, y = make_features(df)
    if len(X) < 200:
        raise RuntimeError("not enough data to train " + tf_key)
    tscv = TimeSeriesSplit(n_splits=TRAIN_SPLITS)
    base = GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=4)
    clf = CalibratedClassifierCV(base, method='isotonic', cv=tscv)
    clf.fit(X, y)
    dump(clf, MODEL_PATHS[tf_key])
    log("Saved model:", MODEL_PATHS[tf_key])
    return clf

def load_or_train_all():
    models = {}
    for tf in TF_MAP:
        if os.path.exists(MODEL_PATHS[tf]):
            log("Loading model for", tf)
            models[tf] = load(MODEL_PATHS[tf])
        else:
            models[tf] = train_model_for_tf(tf)
    return models

def symbol_info():
    info = mt5.symbol_info(SYMBOL)
    if info is None:
        raise RuntimeError("symbol_info failed")
    return info

def equity_now():
    a = mt5.account_info()
    if a is None:
        raise RuntimeError("account_info failed")
    return a.equity

def compute_lots(equity, sl_price_distance_points, info):
    if sl_price_distance_points <= 0:
        return 0.0
    # estimate risk per lot based on simple pip-value fallback
    pipvalue = 10.0  # conservative estimate for 1 standard lot in USD
    risk_per_lot = sl_price_distance_points * pipvalue
    risk_amount = equity * RISK_PER_TRADE
    lots = risk_amount / max(risk_per_lot, 1e-9)
    # conform to broker volume step/limits if available
    vol_step = getattr(info, "volume_step", 0.01)
    vol_min  = getattr(info, "volume_min", 0.01)
    vol_max  = getattr(info, "volume_max", 100.0)
    if vol_step > 0:
        lots = math.floor(lots / vol_step) * vol_step
    lots = max(vol_min, min(vol_max, lots))
    return round(lots, 2)

def close_all_positions():
    pos = mt5.positions_get(symbol=SYMBOL)
    if pos:
        for p in pos:
            typ = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(SYMBOL).ask if typ == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).bid
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": SYMBOL,
                "volume": p.volume,
                "type": typ,
                "price": price,
                "deviation": SLIPPAGE,
                "position": p.ticket,
                "magic": MAGIC,
                "comment": "panic_close"
            }
            mt5.order_send(req)

def daily_guard(start_equity):
    eq = equity_now()
    dd = (start_equity - eq) / start_equity
    if dd >= DAILY_LOSS_CAP:
        log(f"DAILY CAP HIT ({dd:.2%}), closing and halting entries")
        close_all_positions()
        return False
    return True

def send_order(direction, lots, sl_price, tp_price):
    """direction: 'long' or 'short'"""
    price_tick = mt5.symbol_info_tick(SYMBOL)
    if direction == 'long':
        order_type = mt5.ORDER_TYPE_BUY
        price = price_tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = price_tick.bid
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": sl_price,
        "tp": tp_price,
        "deviation": SLIPPAGE,
        "magic": MAGIC,
        "comment": "ml_mtf_" + direction
    }
    res = mt5.order_send(req)
    return res

def live_loop(models):
    info = symbol_info()
    start_equity = equity_now()
    last_m1_time = None
    log("LIVE LOOP START (long & short enabled)")
    while True:
        time.sleep(POLL_SECONDS)
        # poll latest M1 bar
        df_m1 = fetch_rates("M1", 200)
        tnow = df_m1.index[-1]
        if last_m1_time is None:
            last_m1_time = tnow
            continue
        if tnow <= last_m1_time:
            continue
        last_m1_time = tnow

        # daily safety
        if not daily_guard(start_equity):
            continue

        # fetch TF data (small windows)
        try:
            df1 = df_m1
            df5 = fetch_rates("M5", 400)
            df15 = fetch_rates("M15", 200)
        except Exception as e:
            log("fetch error:", e); continue

        # features & model proba for each TF (use last valid row)
        probs = {}
        for tf_key, df in [("M1",df1),("M5",df5),("M15",df15)]:
            X, _ = make_features(df)
            if len(X) < 20:
                probs[tf_key] = 0.5
                continue
            x = X.iloc[-1:].fillna(0)
            m = models.get(tf_key)
            try:
                p = float(m.predict_proba(x)[:,1])
            except Exception as e:
                log("model predict error for", tf_key, e)
                p = 0.5
            probs[tf_key] = p

        avg_proba = np.mean(list(probs.values()))
        votes_long = sum(1 for p in probs.values() if p > TF_THRESH)
        votes_short = sum(1 for p in probs.values() if p < TF_THRESH_SHORT)
        log("probas:", {k:round(v,3) for k,v in probs.items()}, "avg", round(avg_proba,3),
            "votes_long", votes_long, "votes_short", votes_short)

        # determine open positions
        open_pos = mt5.positions_get(symbol=SYMBOL) or []
        has_long = any(p.type == mt5.ORDER_TYPE_BUY for p in open_pos)
        has_short = any(p.type == mt5.ORDER_TYPE_SELL for p in open_pos)

        # compute ATR from M5 for sizing/SL/TP
        try:
            X5, _ = make_features(df5)
            atr = float(X5['atr'].iloc[-1])
            if math.isnan(atr) or atr <= 0:
                continue
        except Exception:
            continue

        # price distances
        sl_price = SL_ATR_MULT * atr
        tp_price = TP_ATR_MULT * atr

        pt = info.point
        sl_points = sl_price / pt

        equity = equity_now()
        lots = compute_lots(equity, sl_points, info)
        if lots < getattr(info, "volume_min", 0.01):
            log("lots too small:", lots, "skip entries this bar")
            continue

        # ENTRY LONG: avg proba > ENTRY_THRESHOLD and >=2 TF votes
        if avg_proba > ENTRY_THRESHOLD and votes_long >= 2 and not has_long:
            # close short if present
            if has_short:
                log("Closing existing short before opening long")
                close_all_positions()
                time.sleep(0.5)
            price = mt5.symbol_info_tick(SYMBOL).ask
            sl = price - sl_price
            tp = price + tp_price
            res = send_order('long', lots, sl, tp)
            log("LONG order send ->", getattr(res, "retcode", None), "lots", lots)

        # ENTRY SHORT: avg_proba < ENTRY_THRESHOLD_SHORT and >=2 short votes
        elif avg_proba < ENTRY_THRESHOLD_SHORT and votes_short >= 2 and not has_short:
            if has_long:
                log("Closing existing long before opening short")
                close_all_positions()
                time.sleep(0.5)
            price = mt5.symbol_info_tick(SYMBOL).bid
            sl = price + sl_price
            tp = price - tp_price
            res = send_order('short', lots, sl, tp)
            log("SHORT order send ->", getattr(res, "retcode", None), "lots", lots)

        # EXIT conditions: take conservative approach using avg proba and vote disappearance
        if has_long and (avg_proba < EXIT_THRESHOLD or votes_long == 0):
            log("Exit condition met for LONG -> closing positions")
            close_all_positions()

        if has_short and (avg_proba > (1.0 - EXIT_THRESHOLD) or votes_short == 0):
            log("Exit condition met for SHORT -> closing positions")
            close_all_positions()

if __name__ == "__main__":
    mt5_init()
    models = load_or_train_all()
    try:
        live_loop(models)
    except KeyboardInterrupt:
        log("stopped by user")
    finally:
        mt5.shutdown()
