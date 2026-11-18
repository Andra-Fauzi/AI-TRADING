"""
================================================================================
LIVE PREDICTION XAUUSD — MT5 (M5, multi-timeframe M15/M30/H1)
================================================================================
File standalone (tidak butuh Jupyter) untuk live prediction, diambil dari
notebook `prediksi_precision70_binary_3_multiframe_BCE_v5...ipynb`.

CARA PAKAI:
1. Pastikan MetaTrader 5 terminal sudah terinstall & LOGIN ke akun broker.
   (Windows only -- package MetaTrader5 tidak jalan di Linux/Mac.)
2. pip install MetaTrader5 pandas numpy torch ta joblib scikit-learn
3. Taruh file berikut di folder yang SAMA dengan script ini:
     - scaler.pkl
     - buy_model_finetuned.pkl   (atau ganti nama di CONFIG di bawah)
     - sell_model_finetuned.pkl
4. Isi CONFIG di bawah (login MT5, symbol, path model).
5. Jalankan:  python live_prediction_mt5.py
6. Stop dengan Ctrl+C (koneksi MT5 akan ditutup otomatis).
================================================================================
"""

import datetime
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ta.volatility import BollingerBands
from ta.trend import ADXIndicator

warnings.filterwarnings("ignore")

try:
    import MetaTrader5 as mt5
except ImportError as e:
    raise ImportError(
        "Package 'MetaTrader5' tidak ditemukan. Install dengan:\n"
        "    pip install MetaTrader5\n"
        "Catatan: package ini HANYA jalan di Windows (butuh terminal MT5 asli)."
    ) from e


# ==============================================================================
# ================================  CONFIG  ===================================
# ==============================================================================

# --- Kredensial MT5 -----------------------------------------------------------
# Kosongkan (None) MT5_PATH/LOGIN/PASSWORD/SERVER kalau terminal MT5 sudah
# terbuka & login manual -- mt5.initialize() tanpa argumen akan connect ke
# instance yang sedang jalan.
MT5_PATH     = None   # cth: r"C:\Program Files\MetaTrader 5\terminal64.exe"
MT5_LOGIN    = None   # cth: 12345678
MT5_PASSWORD = None   # cth: "password_kamu"
MT5_SERVER   = None   # cth: "Exness-MT5Trial"

# --- Symbol & timeframe --------------------------------------------------------
SYMBOL = "XAUUSD"      # sesuaikan dgn nama simbol di broker ("XAUUSD.m", "GOLD", dll)

# --- Path file model/scaler (harus hasil training dari notebook) --------------
SCALER_PATH     = "scaler.pkl"
BUY_MODEL_PATH  = "buy_model_finetuned.pkl"
SELL_MODEL_PATH = "sell_model_finetuned.pkl"

# --- Parameter model & threshold (harus SAMA dengan saat training) ------------
WINDOW_INF = 96      # WINDOW_SIZE saat training
BUY_THR    = 0.70
SELL_THR   = 0.70
MARGIN     = 0.20
CONF_BUY_MIN  = 0.60
CONF_SELL_MIN = 0.55

NATR_MIN_INFERENCE = 0.0005
ADX_MIN             = 0.20

# --- Jumlah candle yang di-fetch tiap siklus (beri margin agar cukup utk
#     rolling window terpanjang di create_features + WINDOW_INF) --------------
N_M5  = 700
N_HTF = 1000

# --- Device ---------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)


# ==============================================================================
# =========================  FEATURE ENGINEERING  ============================
# ==============================================================================

def add_ema(df):
    df["ema20"]  = df["close"].ewm(span=20).mean()
    df["ema50"]  = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()
    df["ema20_dist"]  = (df["close"] - df["ema20"])  / df["ema20"]
    df["ema50_dist"]  = (df["close"] - df["ema50"])  / df["ema50"]
    df["ema200_dist"] = (df["close"] - df["ema200"]) / df["ema200"]
    return df


def add_bb(df):
    bb = BollingerBands(df["close"], window=20)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_middle"] = bb.bollinger_mavg()
    df["bb_position"] = (
        (df["close"] - df["bb_lower"]) /
        (df["bb_upper"] - df["bb_lower"] + 1e-8)
    )
    return df


def add_macd(df):
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"]        = (ema12 - ema26) / df["close"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    return df


def add_rsi(df, period=14):
    delta = df["close"].diff()
    gain = pd.Series(np.where(delta > 0, delta, 0.0), index=df.index)
    loss = pd.Series(np.where(delta < 0, -delta, 0.0), index=df.index)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi_norm"] = (df["rsi"] - 50) / 50
    return df


def add_atr(df, period=14):
    hl = df["high"] - df["low"]
    hc = np.abs(df["high"] - df["close"].shift())
    lc = np.abs(df["low"] - df["close"].shift())
    tr = np.maximum(hl, np.maximum(hc, lc))
    df["atr"] = pd.Series(tr).rolling(period).mean()
    return df


def calculate_volume_profile_rolling(df, window=60, n_bins=10):
    poc_distances = np.zeros(len(df))
    closes = df['close'].values
    volumes = df['tick_volume'].values

    for i in range(window, len(df)):
        window_closes = closes[i - window:i]
        window_volumes = volumes[i - window:i]

        min_p, max_p = window_closes.min(), window_closes.max()
        if max_p == min_p:
            poc_distances[i] = 0
            continue

        bins = np.linspace(min_p, max_p, n_bins + 1)
        bin_indices = np.digitize(window_closes, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        bin_volumes = np.zeros(n_bins)
        for idx, vol in zip(bin_indices, window_volumes):
            bin_volumes[idx] += vol

        poc_bin_idx = np.argmax(bin_volumes)
        poc_price = (bins[poc_bin_idx] + bins[poc_bin_idx + 1]) / 2
        poc_distances[i] = (closes[i] - poc_price) / poc_price

    return poc_distances


def calculate_price_density_rolling(df, window=120, tolerance_atr_mult=0.5):
    density_counts = np.zeros(len(df))
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    atrs = df['atr'].values

    for i in range(window, len(df)):
        current_close = closes[i]
        current_atr = atrs[i]
        buffer = current_atr * tolerance_atr_mult

        window_highs = highs[i - window:i]
        window_lows = lows[i - window:i]

        high_hits = np.sum((window_highs >= current_close - buffer) & (window_highs <= current_close + buffer))
        low_hits = np.sum((window_lows >= current_close - buffer) & (window_lows <= current_close + buffer))

        density_counts[i] = high_hits + low_hits

    return density_counts / (window * 2)


def calculate_daily_anchors(df_intraday):
    daily = df_intraday['close'].resample('D').ohlc()
    daily['high'] = df_intraday['high'].resample('D').max()
    daily['low'] = df_intraday['low'].resample('D').min()
    daily = daily.dropna()

    daily['prev_high'] = daily['high'].shift(1)
    daily['prev_low'] = daily['low'].shift(1)
    daily['prev_close'] = daily['close'].shift(1)

    daily['PP'] = (daily['prev_high'] + daily['prev_low'] + daily['prev_close']) / 3
    daily['R1'] = (2 * daily['PP']) - daily['prev_low']
    daily['S1'] = (2 * daily['PP']) - daily['prev_high']

    anchor_columns = ['prev_high', 'prev_low', 'PP', 'R1', 'S1']
    return daily[anchor_columns]


def generate_anchor_distance_features(df):
    df = df.copy()
    df.index = pd.to_datetime(df['time'], unit='s')
    df['date_key'] = df.index.date

    daily_anchors = calculate_daily_anchors(df)
    daily_anchors.index = daily_anchors.index.date

    df = df.merge(daily_anchors, left_on='date_key', right_index=True, how='left')
    df[['prev_high', 'prev_low', 'PP', 'R1', 'S1']] = df[['prev_high', 'prev_low', 'PP', 'R1', 'S1']].ffill()

    df['prev_day_high_dist'] = (df['close'] - df['prev_high']) / df['prev_high']
    df['prev_day_low_dist']  = (df['close'] - df['prev_low']) / df['prev_low']

    df['pivot_point_dist']   = (df['close'] - df['PP']) / df['PP']
    df['r1_dist']            = (df['close'] - df['R1']) / df['R1']
    df['s1_dist']            = (df['close'] - df['S1']) / df['S1']

    df = df.drop(columns=['date_key', 'prev_high', 'prev_low', 'PP', 'R1', 'S1'])
    return df.fillna(0)


def calculate_volume_zscore(df, window=20):
    roll_mean = df['tick_volume'].rolling(window=window).mean()
    roll_std = df['tick_volume'].rolling(window=window).std()
    volume_zscore = (df['tick_volume'] - roll_mean) / (roll_std + 1e-8)
    return volume_zscore.fillna(0)


def calculate_atr_zscore(df, window=20):
    roll_mean_atr = df['atr'].rolling(window=window).mean()
    roll_std_atr = df['atr'].rolling(window=window).std()
    atr_zscore = (df['atr'] - roll_mean_atr) / (roll_std_atr + 1e-8)
    return atr_zscore.fillna(0)


def calculate_session_volume_zscore(df, window_days=20):
    df['hour_group'] = df.index.hour
    roll_mean = df.groupby('hour_group')['tick_volume'].transform(
        lambda x: x.rolling(window=window_days, min_periods=1).mean())
    roll_std = df.groupby('hour_group')['tick_volume'].transform(
        lambda x: x.rolling(window=window_days, min_periods=1).std())
    session_zscore = (df['tick_volume'] - roll_mean) / (roll_std + 1e-8)
    df = df.drop(columns=['hour_group'])
    return session_zscore.fillna(0)


def create_features(df):
    df["return"] = df["close"].pct_change()
    df["ret_1"]  = df["close"].pct_change(1)
    df["ret_2"]  = df["close"].pct_change(2)
    df["ret_3"]  = df["close"].pct_change(3)
    df["ret_6"]  = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)
    df["ret_24"] = df["close"].pct_change(24)

    df = add_rsi(df)
    df = add_atr(df)
    df = add_ema(df)
    df = add_bb(df)
    df = add_macd(df)

    df["volatility"] = df["return"].rolling(20).std()

    df["body"]       = (df["close"] - df["open"]) / df["close"]
    df["upper_wick"] = (df["high"] - np.maximum(df["open"], df["close"])) / df["close"]
    df["lower_wick"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["close"]

    datetime_col = pd.to_datetime(df["time"], unit="s")
    hour = datetime_col.dt.hour
    day_of_week = datetime_col.dt.dayofweek

    df["sesi_asia"]    = ((hour >= 6) & (hour <= 14)).astype(int)
    df["sesi_eropa"]   = ((hour >= 14) & (hour <= 22)).astype(int)
    df["sesi_amerika"] = ((hour >= 19) | (hour <= 3)).astype(int)
    df["sesi_golden"]  = ((hour >= 19) & (hour <= 22)).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_sin"]  = np.sin(2 * np.pi * day_of_week / 7)
    df["day_cos"]  = np.cos(2 * np.pi * day_of_week / 7)

    df["momentum_3"] = (df["close"] - df["close"].shift(3)) / df["close"]
    df["momentum_6"] = (df["close"] - df["close"].shift(6)) / df["close"]

    df["vol_ma"]    = df["tick_volume"].rolling(10).mean()
    df["vol_ratio"] = df["tick_volume"] / (df["vol_ma"] + 1e-9)

    df["bar_strength"] = np.abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-8)

    inside_sqrt = 0.5 * (np.log(df["high"] / df["low"])) ** 2 - (
        2 * np.log(2) - 1) * (np.log(df["close"] / df["open"])) ** 2
    df["vol_gk"] = np.sqrt(np.clip(inside_sqrt, 0, None))

    vol_m = df["vol_gk"].rolling(window=96).mean()
    vol_s = df["vol_gk"].rolling(window=96).std()
    df["vol_zscore"] = (df["vol_gk"] - vol_m) / (vol_s + 1e-8)

    df["natr"] = df["atr"] / df["close"]

    is_bull = (df["close"] > df["open"]).astype(int)
    is_bear = (df["close"] < df["open"]).astype(int)
    df["bull_streak"] = (is_bull.groupby((is_bull != is_bull.shift()).cumsum()).cumsum()) * is_bull
    df["bear_streak"] = (is_bear.groupby((is_bear != is_bear.shift()).cumsum()).cumsum()) * is_bear

    rolling_min = df["low"].rolling(window=96).min()
    rolling_max = df["high"].rolling(window=96).max()
    df["price_position_96"] = (df["close"] - rolling_min) / (rolling_max - rolling_min + 1e-8)

    df['rsi_slope_3']  = df['rsi'].diff(3)
    df['price_slope']  = df['close'].pct_change(3)
    df['rsi_div']      = df['rsi_slope_3'] - df['price_slope'] * 100

    df['vol_confirm'] = df['vol_ratio'] * df['body']

    df['atr_ratio'] = df['atr'] / df['atr'].rolling(50).mean()

    df['efficiency'] = (df['close'] - df['open']).abs() / \
                       (df['high'] - df['low'] + 1e-8)

    df['trend_1h'] = np.sign(df['close'] - df['close'].ewm(span=12).mean())
    df['trend_4h'] = np.sign(df['close'] - df['close'].ewm(span=48).mean())

    df['poc_dist'] = calculate_volume_profile_rolling(df, window=60, n_bins=10)
    df['price_density'] = calculate_price_density_rolling(df, window=120, tolerance_atr_mult=0.5)

    df = generate_anchor_distance_features(df)

    df['volume_zscore'] = calculate_volume_zscore(df, window=20)
    df['atr_zscore'] = calculate_atr_zscore(df, window=20)
    df['volume_zscore_modular'] = calculate_session_volume_zscore(df, window_days=14)

    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum((df['high'] - df['close'].shift(1)).abs(),
                                     (df['low'] - df['close'].shift(1)).abs()))
    df['atr_14'] = df['tr'].rolling(window=14).mean()

    df['tr_percentage'] = df['tr'] / df['close']
    df['atr_14_percentage'] = df['atr_14'] / df['close']

    df['bullish_fvg_gap'] = df['low'] - df['high'].shift(2)
    df['valid_scalp_fvg'] = ((df['bullish_fvg_gap'] > 0) & (df['bullish_fvg_gap'] > (df['atr_14'] * 0.5))).astype(int)
    df['entry_price'] = np.where(df['valid_scalp_fvg'] == 1, df['high'].shift(1), 0.0)
    df['bullish_sl'] = np.where(df['valid_scalp_fvg'] == 1, df['low'].shift(1), 0.0)
    df['bullish_fvg_dist'] = np.where(df['entry_price'] > 0, (df['close'] - df['entry_price']) / df['close'], 0.0)

    df['bearish_fvg_gap'] = df['low'].shift(2) - df['high']
    df['valid_scalp_bearish_fvg'] = ((df['bearish_fvg_gap'] > 0) & (df['bearish_fvg_gap'] > (df['atr_14'] * 0.5))).astype(int)
    df['bearish_entry_price'] = np.where(df['valid_scalp_bearish_fvg'] == 1, df['low'].shift(1), 0.0)
    df['bearish_stop_loss'] = np.where(df['valid_scalp_bearish_fvg'] == 1, df['high'], 0.0)
    df['bearish_fvg_dist'] = np.where(df['bearish_entry_price'] > 0, (df['close'] - df['bearish_entry_price']) / df['close'], 0.0)

    df["ema20_slope"] = df["ema20"].pct_change(3)
    df["ema50_slope"] = df["ema50"].pct_change(5)
    df["ema200_slope"] = df["ema200"].pct_change(10)

    df["ema20_50"] = (df["ema20"] - df["ema50"]) / df["atr"]
    df["ema50_200"] = (df["ema50"] - df["ema200"]) / df["atr"]

    df["trend_strength"] = abs(df["ema20"] - df["ema50"]) / (df["atr"] + 1e-8)

    lookback = 48
    df["swing_high"] = df["high"].rolling(lookback).max()
    df["swing_low"]  = df["low"].rolling(lookback).min()

    df["dist_high"] = (df["swing_high"] - df["close"]) / (df["atr"] + 1e-8)
    df["dist_low"]  = (df["close"] - df["swing_low"]) / (df["atr"] + 1e-8)

    df["rv12"] = df["return"].rolling(12).std()
    df["rv24"] = df["return"].rolling(24).std()
    df["rv48"] = df["return"].rolling(48).std()

    df["vol_term"] = df["rv12"] / df["rv48"]

    roll = df.close.rolling(48)
    df["z_close"] = (df.close - roll.mean()) / (roll.std() + 1e-8)

    df["trend_regime"] = (df["trend_strength"] > 1.5).astype(int)
    df["vol_regime"] = (df["atr_ratio"] > 1.3).astype(int)
    df["range_regime"] = (df["price_position_96"].between(0.3, 0.7)).astype(int)

    df["rolling_trend"] = (df["ema20"] - df["ema50"]) / (df["atr"] + 1e-8)
    df["return_acceleration"] = df["ret_3"] - df["ret_12"]

    df["rsi_slope"] = df["rsi"].diff()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["swing_range"] = df["dist_high"] + df["dist_low"]

    _prev_vol = df["tick_volume"].shift(1)
    df["volume_change"] = ((df["tick_volume"] - _prev_vol) / (_prev_vol.abs() + 1e-9)).clip(-20, 20)

    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx()
    df["+di"] = adx.adx_pos()
    df["-di"] = adx.adx_neg()
    df["adx"] = adx.adx()
    df["plus_di"] = adx.adx_pos()
    df["minus_di"] = adx.adx_neg()

    df["ret_48"] = df["close"].pct_change(48)
    df["ret_96"] = df["close"].pct_change(96)

    df["trend_persistence"] = (
        (df["close"] > df["ema20"]).astype(int) +
        (df["ema20"] > df["ema50"]).astype(int) +
        (df["ema50"] > df["ema200"]).astype(int)
    )

    highest48 = df["high"].rolling(48).max()
    lowest48  = df["low"].rolling(48).min()
    df["dist_highest48"] = (highest48 - df["close"]) / df["atr"]
    df["dist_lowest48"]  = (df["close"] - lowest48) / df["atr"]

    df["atr_expand"] = df["atr"] / df["atr"].shift(10)

    df["ema_compression"] = (df["ema20"] - df["ema50"]).abs() / df["atr"]

    df["rsi_velocity"] = df["rsi"].diff()
    df["rsi_acceleration"] = df["rsi_velocity"].diff()

    bb_width = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_width"] = bb_width
    df["bb_width_change"] = bb_width.pct_change()

    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / range_
    df["bull_strength"] = (df["close"] - df["low"]) / range_
    df["bear_strength"] = (df["high"] - df["close"]) / range_

    df["volatility_rank"] = df["volatility"].rolling(200).rank(pct=True)

    highest20 = df["high"].rolling(20).max()
    lowest20  = df["low"].rolling(20).min()
    df["breakout_up"] = (df["close"] - highest20.shift(1)) / df["atr"]
    df["breakout_down"] = (lowest20.shift(1) - df["close"]) / df["atr"]

    highest200 = df["high"].rolling(200).max()
    lowest200  = df["low"].rolling(200).min()
    df["price_position200"] = (df["close"] - lowest200) / (highest200 - lowest200)

    df["volume_momentum"] = df["tick_volume"] / df["tick_volume"].rolling(20).mean()

    df["momentum_acc"] = df["ret_12"] - df["ret_48"]

    df = df.dropna(subset=['volatility', 'vol_zscore', 'atr_ratio', 'price_position_96'])
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df.copy(deep=True)

    return df


# --- Higher-timeframe feature merge (anti-lookahead) --------------------------
HTF_FEATURES = [
    'ema20_dist', 'ema50_dist', 'ema200_dist',
    'ema20_slope', 'ema50_slope', 'ema200_slope',
    'trend_strength', 'rsi_norm', 'rsi_slope',
    'macd', 'macd_signal', 'macd_hist',
    'natr', 'atr_ratio', 'bb_position',
    'adx', 'plus_di', 'minus_di',
]


def prep_htf(df_htf, prefix, shift_minutes):
    """Geser timestamp HTF maju 1 periode penuh supaya candle yang dipakai
    SUDAH benar-benar closed sebelum digabung ke M5 (anti-lookahead)."""
    d = df_htf.copy()
    d["time_dt"] = (
        pd.to_datetime(d["time"], unit="s") + pd.Timedelta(minutes=shift_minutes)
    ).astype("datetime64[ns]")
    cols = ["time_dt"] + HTF_FEATURES
    d = d[cols].add_prefix(prefix)
    d = d.rename(columns={f"{prefix}time_dt": "time_dt"})
    return d.sort_values("time_dt")


# --- Urutan fitur INPUT MODEL (harus identik dengan urutan saat training) ----
FEATURE_COLUMNS = [
    'return', 'ret_2', 'ret_3', 'ret_6', 'ret_12', 'ret_24', 'return_acceleration',
    'ema20_dist', 'ema50_dist', 'ema200_dist',
    'ema20_slope', 'ema50_slope', 'ema200_slope',
    'ema20_50', 'ema50_200',
    'trend_strength',
    'rsi_norm', 'rsi_slope',
    'macd', 'macd_signal', 'macd_hist',
    'natr',
    'volatility', 'vol_gk',
    'atr_ratio', 'atr_zscore',
    'rv12', 'rv24', 'rv48',
    'vol_term',
    'tr_percentage',
    'price_position_96',
    'dist_high', 'dist_low',
    'swing_range',
    'z_close',
    'bb_position',
    'body', 'upper_wick', 'lower_wick',
    'bar_strength',
    'vol_ratio', 'vol_confirm',
    'volume_zscore', 'volume_zscore_modular',
    'poc_dist', 'price_density',
    'rsi_div',
    'valid_scalp_fvg', 'valid_scalp_bearish_fvg',
    'bullish_fvg_dist', 'bearish_fvg_dist',
    'sesi_asia', 'sesi_eropa', 'sesi_amerika', 'sesi_golden',
    'hour_sin', 'hour_cos',
    'day_sin', 'day_cos',
    'adx', 'plus_di', 'minus_di',
    'ret_48', 'ret_96',
    'trend_persistence',
    'atr_expand',
    'rsi_acceleration',
    'bb_width', 'bb_width_change',
    'body_ratio', 'bull_strength', 'bear_strength',
    'volatility_rank',
    'breakout_up', 'breakout_down',
    'price_position200',
    'volume_momentum',
    'momentum_acc',
] + [f"m15_{f}" for f in HTF_FEATURES] \
  + [f"m30_{f}" for f in HTF_FEATURES] \
  + [f"h1_{f}"  for f in HTF_FEATURES]

N_FEATURES = len(FEATURE_COLUMNS)
print(f"Jumlah fitur: {N_FEATURES}")


# ==============================================================================
# ==============================  MODEL ARCH  =================================
# ==============================================================================
# NOTE: definisi class di bawah HARUS identik dengan saat training, karena
# joblib.load() akan unpickle objek model penuh dan butuh class-nya tersedia
# di namespace ini.

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class PositionalEmbedding(nn.Module):
    def __init__(self, num_patches, d_model):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, input_dim, seq_len, patch_len=16, stride=8, d_model=512, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = ((seq_len - patch_len) // stride) + 1
        self.proj = nn.Linear(patch_len * input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        patches = []
        for start in range(0, T - self.patch_len + 1, self.stride):
            patch = x[:, start:start + self.patch_len, :]
            patch = patch.reshape(B, -1)
            patches.append(patch)
        patches = torch.stack(patches, dim=1)
        patches = self.proj(patches)
        patches = self.norm(patches)
        patches = self.dropout(patches)
        return patches


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.drop_path1 = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    def forward(self, x):
        attn = self.score(x)
        attn = torch.softmax(attn, dim=1)
        feat = (attn * x).sum(dim=1)
        return feat


class PatchTSTEncoder(nn.Module):
    def __init__(self, input_dim, seq_len, patch_len=16, stride=8, d_model=512,
                 depth=4, num_heads=8, mlp_ratio=4, dropout=0.1, drop_path=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            input_dim=input_dim, seq_len=seq_len, patch_len=patch_len,
            stride=stride, d_model=d_model, dropout=dropout,
        )
        self.pos_embed = PositionalEmbedding(self.patch_embed.num_patches, d_model)
        dpr = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio,
                              dropout=dropout, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.pool = AttentionPooling(d_model)
        self.norm = nn.LayerNorm(d_model)
        self.apply(init_weights)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        feat = self.pool(x)
        feat = self.norm(feat)
        return feat


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(channels, channels // reduction), nn.GELU(),
            nn.Linear(channels // reduction, channels), nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x).unsqueeze(-1)
        return x * scale


class ResidualSEBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=1), nn.BatchNorm1d(channels),
        )
        self.se = SEBlock(channels)
        self.act = nn.GELU()

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return self.act(out + x)


class RecencyGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.drop(self.norm(last))


class HighPrecisionTradingModelAttnPatch(nn.Module):
    def __init__(self, n_features=57, n_classes=1, width=512, dropout=0.4,
                 seq_len=128, patch_len=16, stride=8, depth=4, num_heads=8):
        super().__init__()
        half = width // 2
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, half, kernel_size=3, padding=1), nn.BatchNorm1d(half), nn.GELU(),
            ResidualSEBlock(half, dropout * 0.5),
            nn.Conv1d(half, width, kernel_size=5, padding=2), nn.BatchNorm1d(width), nn.GELU(),
            ResidualSEBlock(width, dropout * 0.5),
            nn.Dropout(dropout),
        )
        self.patchtst = PatchTSTEncoder(
            input_dim=width, seq_len=seq_len, patch_len=patch_len, stride=stride,
            d_model=width, depth=depth, num_heads=num_heads, mlp_ratio=4,
            dropout=dropout, drop_path=0.1,
        )
        self.class_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width // 2, width // 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width // 4, n_classes),
        )
        self.conf_head = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width // 2, 1),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        feat = self.patchtst(x)
        logits = self.class_head(feat)
        conf = self.conf_head(feat)
        return logits, conf


# ==============================================================================
# ==========================  MT5 INIT & FETCH DATA  ===========================
# ==============================================================================

def init_mt5():
    kwargs = {}
    if MT5_PATH:
        kwargs["path"] = MT5_PATH
    if MT5_LOGIN:
        kwargs["login"] = MT5_LOGIN
    if MT5_PASSWORD:
        kwargs["password"] = MT5_PASSWORD
    if MT5_SERVER:
        kwargs["server"] = MT5_SERVER

    ok = mt5.initialize(**kwargs)
    if not ok:
        raise RuntimeError(f"MT5 initialize() gagal, error = {mt5.last_error()}")

    acc = mt5.account_info()
    term = mt5.terminal_info()
    if acc is not None and term is not None:
        print(f"✅ Terhubung ke MT5 — Terminal: {term.name} | "
              f"Akun: {acc.login} ({acc.server}) | Balance: {acc.balance}")
    else:
        print("✅ MT5 initialize() sukses (info akun/terminal belum tersedia).")


def fetch_mt5_candles(timeframe, n_candles):
    """Ambil n_candles terakhir langsung dari MT5 (posisi 0 = candle paling
    baru, yang BISA JADI masih running/belum closed -- ditangani di
    get_latest_data() dengan cara buang baris terakhir)."""
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, n_candles)
    if rates is None or len(rates) == 0:
        raise ValueError(f"Gagal ambil data MT5 timeframe={timeframe} (error: {mt5.last_error()})")
    df = pd.DataFrame(rates)
    df["time"] = df["time"].astype("int64")
    for col in ("open", "high", "low", "close"):
        df[col] = df[col].astype("float32")
    df["tick_volume"] = pd.to_numeric(df["tick_volume"], downcast="integer")
    return df[["time", "open", "high", "low", "close", "tick_volume"]]


def get_latest_data(n_m5=N_M5, n_htf=N_HTF):
    df_live = fetch_mt5_candles(TF_M5, n_m5 + 5)
    df_live = create_features(df_live)
    if len(df_live) < WINDOW_INF:
        raise ValueError(f"Data M5 terlalu sedikit: {len(df_live)}")

    df_live["time_dt"] = pd.to_datetime(df_live["time"], unit="s").astype("datetime64[ns]")
    df_live = df_live.sort_values("time_dt")
    df_live = df_live.tail(n_m5).reset_index(drop=True)

    m15_raw = fetch_mt5_candles(TF_M15, n_htf + 5)
    m15_sub = prep_htf(create_features(m15_raw), "m15_", shift_minutes=15)

    m30_raw = fetch_mt5_candles(TF_M30, n_htf + 5)
    m30_sub = prep_htf(create_features(m30_raw), "m30_", shift_minutes=30)

    h1_raw = fetch_mt5_candles(TF_H1, n_htf + 5)
    h1_sub = prep_htf(create_features(h1_raw), "h1_", shift_minutes=60)

    df_live = pd.merge_asof(df_live, m15_sub, on="time_dt", direction="backward")
    df_live = pd.merge_asof(df_live, m30_sub, on="time_dt", direction="backward")
    df_live = pd.merge_asof(df_live, h1_sub,  on="time_dt", direction="backward")
    df_live = df_live.dropna()

    if df_live.empty:
        raise ValueError("Setelah merge HTF & dropna, DataFrame kosong.")
    if len(df_live) < WINDOW_INF + 1:
        raise ValueError(f"Tidak cukup data ({len(df_live)} rows) untuk WINDOW_INF+1.")

    # Buang candle terakhir -- itu candle M5 yang masih running/belum closed
    return df_live.copy()[:-1]


def prepare_input(df_live):
    feats = df_live[FEATURE_COLUMNS].values[-WINDOW_INF:]
    feats = scaler_X.transform(feats)
    tensor = torch.tensor(feats.reshape(1, WINDOW_INF, N_FEATURES), dtype=torch.float32).to(device)
    return tensor


# ==============================================================================
# ================================  PREDIKSI  ==================================
# ==============================================================================

def predict_v2():
    # Auto-reconnect kalau koneksi MT5 putus
    if mt5.terminal_info() is None:
        print("⚠️ Koneksi MT5 putus, reconnect...")
        init_mt5()
        mt5.symbol_select(SYMBOL, True)

    df_live = get_latest_data()

    last = df_live.iloc[-1]
    last_natr = last["natr"]
    last_adx = last.get("adx", last.get("atr_ratio", 99))

    if last_natr < NATR_MIN_INFERENCE:
        print(f"⏸ SKIP — NATR={last_natr:.5f} terlalu rendah (sideways)")
        return None, 0

    if last_adx < ADX_MIN:
        print(f"⏸ SKIP — ADX={last_adx:.2f} terlalu rendah (tidak trending)")
        return None, 0

    x = prepare_input(df_live)

    with torch.no_grad():
        logits_buy, conf_buy = inf_buy_model(x)
        logits_sell, conf_sell = inf_sell_model(x)

    prob_buy = torch.sigmoid(logits_buy)[0, 0].item()
    prob_sell = torch.sigmoid(logits_sell)[0, 0].item()
    conf_b = torch.sigmoid(conf_buy).item()
    conf_s = torch.sigmoid(conf_sell).item()

    print(f"📊 prob_buy={prob_buy:.3f} (conf={conf_b:.3f})  "
          f"prob_sell={prob_sell:.3f} (conf={conf_s:.3f})")

    signal_buy = (prob_buy > BUY_THR) and (conf_b > CONF_BUY_MIN)
    signal_sell = (prob_sell > SELL_THR) and (conf_s > CONF_SELL_MIN)

    if abs(prob_buy - prob_sell) < MARGIN:
        print(f"⚪ HOLD — margin tipis ({abs(prob_buy - prob_sell):.3f} < {MARGIN})")
        return 0, max(prob_buy, prob_sell)

    if signal_buy and signal_sell:
        signal_buy = prob_buy >= prob_sell
        signal_sell = not signal_buy

    if signal_buy:
        print(f"📈 BUY  — prob={prob_buy:.3f}  conf={conf_b:.3f}")
        return 1, prob_buy
    elif signal_sell:
        print(f"📉 SELL — prob={prob_sell:.3f}  conf={conf_s:.3f}")
        return 2, prob_sell
    else:
        print(f"🟡 HOLD — prob_buy={prob_buy:.3f} prob_sell={prob_sell:.3f}")
        return 0, max(prob_buy, prob_sell)


# ==============================================================================
# =====================  LOOP — PREDICT TIAP CANDLE M5 BARU  ==================
# ==============================================================================

LABELS = {0: "🟡 HOLD", 1: "📈 BUY", 2: "📉 SELL"}


def get_last_closed_candle_time():
    """Posisi index=1 di copy_rates_from_pos = candle M5 terakhir yang
    SUDAH closed (index=0 selalu candle yang masih running)."""
    rates = mt5.copy_rates_from_pos(SYMBOL, TF_M5, 1, 1)
    if rates is None or len(rates) == 0:
        return None
    return pd.to_datetime(rates[0]["time"], unit="s")


def seconds_until_next_m5_close(buffer_sec=3):
    now = datetime.datetime.utcnow()
    minutes_to_next = 5 - (now.minute % 5)
    if minutes_to_next == 0:
        minutes_to_next = 5
    next_close = now.replace(second=0, microsecond=0) + datetime.timedelta(minutes=minutes_to_next)
    return max((next_close - now).total_seconds() + buffer_sec, buffer_sec)


def run_live_loop():
    last_candle_time = None
    index_i = 1

    print("🚀 Live prediction dimulai — hanya predict tiap candle M5 baru (MT5)...\n")

    try:
        while True:
            try:
                wait_sec = seconds_until_next_m5_close()
                print(f"⏳ Menunggu {wait_sec:.0f}s sampai candle M5 berikutnya close...")
                time.sleep(wait_sec)

                current_candle_time = get_last_closed_candle_time()
                if current_candle_time is None:
                    print("⚠️ Gagal ambil candle dari MT5, retry 5s...")
                    time.sleep(5)
                    continue

                if last_candle_time is not None and current_candle_time <= last_candle_time:
                    print(f"↩️  Candle belum baru ({current_candle_time}), retry 5s...")
                    time.sleep(5)
                    continue

                last_candle_time = current_candle_time
                signal, confidence = predict_v2()
                if signal is not None:
                    print(f"#{index_i} | candle {current_candle_time} | "
                          f"{LABELS[signal]} | conf={confidence * 100:.2f}%\n")
                index_i += 1

            except Exception as e:
                print(f"⚠️ Error: {e}")
                time.sleep(5)

    except KeyboardInterrupt:
        print("\n🛑 Dihentikan manual — menutup koneksi MT5...")
        mt5.shutdown()


# ==============================================================================
# =================================  MAIN  =====================================
# ==============================================================================

if __name__ == "__main__":
    # --- Init MT5 & pilih symbol -------------------------------------------
    init_mt5()

    if not mt5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"Symbol {SYMBOL} tidak ditemukan / gagal di-select di Market Watch")

    TF_M5  = mt5.TIMEFRAME_M5
    TF_M15 = mt5.TIMEFRAME_M15
    TF_M30 = mt5.TIMEFRAME_M30
    TF_H1  = mt5.TIMEFRAME_H1

    # --- Load scaler & model -------------------------------------------------
    scaler_X = joblib.load(SCALER_PATH)
    inf_buy_model = joblib.load(BUY_MODEL_PATH).to(device)
    inf_sell_model = joblib.load(SELL_MODEL_PATH).to(device)
    inf_buy_model.eval()
    inf_sell_model.eval()
    print(f"✅ Model & scaler ter-load ({BUY_MODEL_PATH}, {SELL_MODEL_PATH}, {SCALER_PATH})")

    # --- Mulai loop live prediction -----------------------------------------
    run_live_loop()