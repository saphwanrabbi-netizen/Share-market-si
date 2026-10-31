import io
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import cv2
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ============================================================
# Configuration
# ============================================================
FEATURES = [
    "ret_1d", "ret_5d", "ret_20d",
    "sma20", "sma50", "ema12",
    "rsi14", "macd", "macd_signal", "macd_hist",
    "vol_change", "vol_ratio_20",
    "sma20_gap", "sma50_gap",
    "high_low_range", "close_position"
]

TRANSACTION_COST = 0.002
MIN_TEST_ROWS = 30

st.set_page_config(
    page_title="Share Market AI",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# Validation
# ============================================================
def validate_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("Ticker symbol লিখুন।")

    if len(ticker) > 15 or any(ch in ticker for ch in " /\\?=&"):
        raise ValueError("Ticker symbol সঠিক নয়।")

    return ticker


def validate_start_date(start: str) -> str:
    try:
        parsed = pd.to_datetime(start)
    except Exception:
        raise ValueError("তারিখ সঠিক নয়। যেমন: 2015-01-01")

    if parsed > pd.Timestamp.today() - pd.Timedelta(days=120):
        raise ValueError("শুরুর তারিখ কমপক্ষে ১২০ দিন পুরনো হতে হবে।")

    return parsed.strftime("%Y-%m-%d")


# ============================================================
# Data functions
# ============================================================
def download_price_data(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if df is None or df.empty:
        raise ValueError(f"{ticker}: কোনো ডেটা পাওয়া যায়নি। Ticker symbol দেখুন।")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"প্রয়োজনীয় কলাম পাওয়া যায়নি: {', '.join(missing)}")

    df = df[required].copy()
    df = df.dropna()

    if len(df) < 120:
        raise ValueError("পর্যাপ্ত historical data নেই। অন্য ticker বা পুরনো start date দিন।")

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    df["ret_1d"] = close.pct_change(1)
    df["ret_5d"] = close.pct_change(5)
    df["ret_20d"] = close.pct_change(20)

    df["sma20"] = close.rolling(20).mean()
    df["sma50"] = close.rolling(50).mean()
    df["ema12"] = close.ewm(span=12, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["vol_change"] = volume.pct_change(1)
    df["vol_ratio_20"] = volume / (volume.rolling(20).mean() + 1e-9)

    df["sma20_gap"] = (close - df["sma20"]) / (df["sma20"] + 1e-9)
    df["sma50_gap"] = (close - df["sma50"]) / (df["sma50"] + 1e-9)

    df["high_low_range"] = (high - low) / (close + 1e-9)
    df["close_position"] = (close - low) / (high - low + 1e-9)

    df["swing_high_20"] = high.rolling(20).max()
    df["swing_low_20"] = low.rolling(20).min()

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def add_target(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    df["future_return"] = df["Close"].pct_change(horizon).shift(-horizon)
    df["target"] = (df["future_return"] > 0).astype(int)

    df = df.dropna(subset=FEATURES + ["target"])

    if df.empty:
        raise ValueError("পর্যাপ্ত ডেটা নেই। শুরুর তারিখ আরও পুরনো করুন।")

    return df


# ============================================================
# Cached functions
# ============================================================
@st.cache_data(ttl=1800, show_spinner=False)
def load_prepared_data(ticker: str, start: str) -> pd.DataFrame:
    df = download_price_data(ticker, start)
    df = add_indicators(df)
    df = add_target(df)
    return df


@st.cache_resource(show_spinner=False, max_entries=10)
def train_model_cached(
    ticker: str,
    start: str,
    test_ratio: float,
    n_estimators: int
):
    df = load_prepared_data(ticker, start)

    X = df[FEATURES].values
    y = df["target"].values

    split_idx = int(len(X) * (1 - test_ratio))

    if len(X) - split_idx < MIN_TEST_ROWS:
        raise ValueError("Test data খুব ছোট। Test ratio কমান বা আরও পুরনো তারিখ দিন।")

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model, df, split_idx, X_test, y_test


# ============================================================
# Backtest
# ============================================================
def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
    running_max = cumulative_returns.cummax()
    drawdown = cumulative_returns / running_max - 1
    return float(drawdown.min())


def backtest(df, y_pred, y_prob, split_idx, threshold, cost=TRANSACTION_COST):
    test_df = df.iloc[split_idx:].copy()
    test_df["probability"] = y_prob
    test_df["signal"] = (y_prob >= threshold).astype(int)
    test_df["position"] = test_df["signal"].shift(1).fillna(0)

    test_df["strategy_return"] = test_df["position"] * test_df["ret_1d"]

    position_change = test_df["position"].diff().abs().fillna(0)
    test_df["net_return"] = test_df["strategy_return"] - (position_change * cost)

    test_df["cum_strategy"] = (1 + test_df["net_return"]).cumprod()
    test_df["cum_buyhold"] = (1 + test_df["ret_1d"]).cumprod()

    active_days = test_df["position"] != 0
    strategy_returns = test_df.loc[active_days, "net_return"]

    sharpe = 0.0
    if strategy_returns.std() > 0:
        sharpe = float(
            strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        )

    winning_trades = test_df.loc[
        (test_df["position"].shift(1) == 1) & (test_df["ret_1d"] > 0)
    ]

    return {
        "test_df": test_df,
        "strategy_return": float(test_df["cum_strategy"].iloc[-1] - 1),
        "buyhold_return": float(test_df["cum_buyhold"].iloc[-1] - 1),
        "max_drawdown": calculate_max_drawdown(test_df["cum_strategy"]),
        "sharpe": sharpe,
        "trade_count": int(test_df["position"].diff().abs().sum()),
        "win_rate": float(len(winning_trades) / max(active_days.sum(), 1))
    }


# ============================================================
# Smart Money Concepts
# ============================================================
def analyze_smart_money(df, lookback=60):
    """
    সহজ SMC logic:
    - Bullish BOS: সাম্প্রতিক swing high ভাঙলে
    - Bearish BOS: সাম্প্রতিক swing low ভাঙলে
    - Liquidity sweep: আগের high/low ছুঁয়ে ফিরে আসা
    - Volume confirmation
    """
    recent = df.tail(lookback)
    close = recent["Close"]
    high = recent["High"]
    low = recent["Low"]
    volume = recent["Volume"]

    prior_high = high.iloc[:-5].max()
    prior_low = low.iloc[:-5].min()
    latest_close = close.iloc[-1]
    latest_volume_ratio = df["vol_ratio_20"].iloc[-1]

    score = 0.0
    notes = []

    if latest_close > prior_high:
        score += 0.35
        notes.append("Bullish Break of Structure (BOS)")
    elif latest_close < prior_low:
        score -= 0.35
        notes.append("Bearish Break of Structure (BOS)")

    recent_low = low.iloc[-5:].min()
    recent_high = high.iloc[-5:].max()

    if recent_low < prior_low and latest_close > prior_low:
        score += 0.25
        notes.append("সম্ভাব্য bullish liquidity sweep")

    if recent_high > prior_high and latest_close < prior_high:
        score -= 0.25
        notes.append("সম্ভাব্য bearish liquidity sweep")

    if latest_volume_ratio > 1.3:
        if score > 0:
            score += 0.15
            notes.append("Volume দিয়ে bullish move confirm")
        elif score < 0:
            score -= 0.15
            notes.append("Volume দিয়ে bearish move confirm")

    score = max(-1, min(1, score))

    if score > 0.15:
        label = "Bullish SMC"
    elif score < -0.15:
        label = "Bearish SMC"
    else:
        label = "Neutral SMC"

    return {
        "label": label,
        "score": score,
        "notes": notes,
        "prior_high": prior_high,
        "prior_low": prior_low
    }


# ============================================================
# Wyckoff analysis
# ============================================================
def analyze_wyckoff(df, lookback=80):
    """
    সহজ Wyckoff logic:
    - Accumulation: range-bound market, volume drying up, পরে bullish breakout
    - Distribution: range-bound market, volume কমে আসা rally, পরে bearish breakdown
    """
    recent = df.tail(lookback)
    close = recent["Close"]
    volume = recent["Volume"]

    range_high = recent["High"].max()
    range_low = recent["Low"].min()
    latest_close = close.iloc[-1]

    price_range = (range_high - range_low) / (range_high + 1e-9)

    first_half_vol = volume.iloc[:len(volume) // 2].mean()
    second_half_vol = volume.iloc[len(volume) // 2:].mean()

    volume_trend = (second_half_vol - first_half_vol) / (first_half_vol + 1e-9)

    recent_return = close.iloc[-1] / close.iloc[-10] - 1

    score = 0.0
    notes = []

    if price_range < 0.18:
        notes.append("Price একটি range/trading zone-এ আছে")

        if latest_close > (range_high + range_low) / 2 and recent_return > 0:
            score += 0.30
            notes.append("Range-এর উপরের দিকে শক্তি; সম্ভাব্য accumulation")

        if latest_close < (range_high + range_low) / 2 and recent_return < 0:
            score -= 0.30
            notes.append("Range-এর নিচের দিকে দুর্বলতা; সম্ভাব্য distribution")

        if volume_trend < -0.10:
            score += 0.10
            notes.append("Volume কমছে; selling pressure কমতে পারে")

    if latest_close > range_high:
        score += 0.35
        notes.append("Range high ভাঙছে; সম্ভাব্য markup")
    elif latest_close < range_low:
        score -= 0.35
        notes.append("Range low ভাঙছে; সম্ভাব্য markdown")

    score = max(-1, min(1, score))

    if score > 0.15:
        label = "Bullish Wyckoff"
    elif score < -0.15:
        label = "Bearish Wyckoff"
    else:
        label = "Neutral Wyckoff"

    return {
        "label": label,
        "score": score,
        "notes": notes,
        "range_high": range_high,
        "range_low": range_low
    }


# ============================================================
# Gann Square of 9
# ============================================================
def gann_levels(price, scale=1.0):
    """
    Gann Square of 9 থেকে সহজ support/resistance projection।
    45°, 90°, 135°, 180° স্তর ব্যবহার করা হয়েছে।
    """
    root = math.sqrt(max(price * scale, 1e-9))

    levels = {}

    for angle in [45, 90, 135, 180, 225, 270, 315, 360]:
        step = angle / 180
        resistance = ((root + step) ** 2) / scale
        support = ((root - step) ** 2) / scale
        levels[f"{angle}° resistance"] = resistance
        levels[f"{angle}° support"] = support

    return levels


def analyze_gann(df):
    price = float(df["Close"].iloc[-1])
    levels = gann_levels(price)

    resistances = sorted([
        value for key, value in levels.items()
        if "resistance" in key and value > price
    ])

    supports = sorted([
        value for key, value in levels.items()
        if "support" in key and value < price
    ], reverse=True)

    next_resistance = resistances[0] if resistances else price * 1.05
    next_support = supports[0] if supports else price * 0.95

    distance_to_resistance = (next_resistance - price) / price
    distance_to_support = (price - next_support) / price

    if distance_to_support < distance_to_resistance:
        score = 0.20
        label = "Support-এর কাছাকাছি"
    else:
        score = -0.20
        label = "Resistance-এর কাছাকাছি"

    return {
        "label": label,
        "score": score,
        "price": price,
        "next_support": next_support,
        "next_resistance": next_resistance,
        "levels": levels
    }


# ============================================================
# Elliott Wave analysis
# ============================================================
def find_swing_points(series, window=5):
    highs = []
    lows = []

    for i in range(window, len(series) - window):
        current = series.iloc[i]

        if current == series.iloc[i - window:i + window + 1].max():
            highs.append(i)

        if current == series.iloc[i - window:i + window + 1].min():
            lows.append(i)

    return highs, lows


def analyze_elliott_wave(df, lookback=120):
    """
    সহজ Elliott Wave logic:
    - সাম্প্রতিক swing high/low থেকে impulse বা correction শনাক্ত করার চেষ্টা
    - Elliott-এ impulse সাধারণত 5-wave এবং correction 3-wave ধরা হয়।
    """
    recent = df.tail(lookback)
    close = recent["Close"].reset_index(drop=True)

    highs, lows = find_swing_points(close, window=5)

    if len(highs) < 2 or len(lows) < 2:
        return {
            "label": "Neutral Wave",
            "score": 0.0,
            "notes": ["পর্যাপ্ত swing point পাওয়া যায়নি"]
        }

    last_high = close.iloc[highs[-1]]
    last_low = close.iloc[lows[-1]]
    latest_close = close.iloc[-1]

    wave_range = abs(last_high - last_low)
    position_in_range = (latest_close - last_low) / (wave_range + 1e-9)

    score = 0.0
    notes = []

    if last_high > last_low:
        if position_in_range > 0.70:
            score += 0.30
            notes.append("দাম swing high-এর কাছাকাছি; সম্ভাব্য wave 5 বা distribution zone")

        elif position_in_range < 0.30:
            score += 0.15
            notes.append("দাম swing low-এর কাছাকাছি; সম্ভাব্য corrective zone")

        else:
            notes.append("দাম একটি wave range-এর মধ্যে আছে")

    if latest_close > last_high:
        score += 0.25
        notes.append("সম্ভাব্য bullish impulse continuation")

    if latest_close < last_low:
        score -= 0.25
        notes.append("সম্ভাব্য bearish impulse continuation")

    score = max(-1, min(1, score))

    if score > 0.15:
        label = "Bullish Elliott Wave"
    elif score < -0.15:
        label = "Bearish Elliott Wave"
    else:
        label = "Neutral Elliott Wave"

    return {
        "label": label,
        "score": score,
        "notes": notes,
        "last_high": last_high,
        "last_low": last_low
    }


# ============================================================
# Chart image analysis
# ============================================================
def analyze_chart_image(uploaded_file):
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        raise ValueError("ছবিটি পড়া যায়নি। PNG বা JPG ছবি আপলোড করুন।")

    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    height, width = img.shape[:2]

    if width > 1000:
        scale = 1000 / width
        img = cv2.resize(img, (1000, int(height * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    h, w = edges.shape
    chart_area = edges[int(h * 0.10):int(h * 0.85), :]

    lines = cv2.HoughLinesP(
        chart_area,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=int(w * 0.15),
        maxLineGap=10
    )

    trend_score = 0.0
    detected_lines = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if abs(x2 - x1) < 20:
                continue

            slope = (y2 - y1) / (x2 - x1 + 1e-9)

            if slope < -0.15:
                trend_score += 1
            elif slope > 0.15:
                trend_score -= 1

            detected_lines += 1

    if detected_lines == 0:
        label = "Neutral"
        image_score = 0.0
    elif trend_score > 0:
        label = "Bullish"
        image_score = min(trend_score / 5, 1.0)
    elif trend_score < 0:
        label = "Bearish"
        image_score = max(trend_score / 5, -1.0)
    else:
        label = "Neutral"
        image_score = 0.0

    return {
        "original_rgb": np.array(image),
        "processed_rgb": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        "edges": edges,
        "label": label,
        "score": image_score,
        "detected_lines": detected_lines
    }


def combine_all_predictions(
    ml_probability,
    smc_score,
    wyckoff_score,
    gann_score,
    elliott_score,
    image_score=None,
    image_weight=0.15
):
    weights = {
        "ml": 0.40,
        "smc": 0.20,
        "wyckoff": 0.15,
        "gann": 0.10,
        "elliott": 0.15
    }

    ml_score = (ml_probability - 0.5) * 2

    combined = (
        weights["ml"] * ml_score
        + weights["smc"] * smc_score
        + weights["wyckoff"] * wyckoff_score
        + weights["gann"] * gann_score
        + weights["elliott"] * elliott_score
    )

    if image_score is not None:
        combined = (1 - image_weight) * combined + image_weight * image_score

    combined_probability = (combined + 1) / 2

    if combined_probability >= 0.60:
        signal = "Bullish — সম্ভাব্য ঊর্ধ্বমুখী"
    elif combined_probability <= 0.40:
        signal = "Bearish — সম্ভাব্য নিম্নমুখী"
    else:
        signal = "Neutral — স্পষ্ট signal নেই"

    return combined_probability, signal


# ============================================================
# UI
# ============================================================
st.title("📈 Smart Money Market Analysis ও Prediction AI")
st.caption("SMC • Wyckoff • Gann • Elliott Wave • Machine Learning — শিক্ষামূলক বিশ্লেষণ, guaranteed profit নয়।")

popular = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "NVDA", "META", "RELIANCE.NS", "TCS.NS"
]

col_input1, col_input2 = st.columns(2)

with col_input1:
    ticker = st.selectbox(
        "জনপ্রিয় টিকার বেছে নিন",
        options=popular,
        index=0
    )

with col_input2:
    custom_ticker = st.text_input(
        "অথবা নিজের ticker symbol লিখুন",
        placeholder="যেমন: AAPL, TSLA, RELIANCE.NS",
        help="মোবাইল থেকেও লিখতে পারবেন"
    )

ticker = validate_ticker(custom_ticker) if custom_ticker else ticker

col_settings1, col_settings2, col_settings3 = st.columns(3)

with col_settings1:
    start = st.text_input("শুরুর তারিখ", value="2015-01-01")

with col_settings2:
    test_ratio = st.slider("Test data ratio", 0.1, 0.4, 0.2, 0.05)

with col_settings3:
    n_estimators = st.slider("Model trees", 100, 500, 150, 50)

threshold = st.slider(
    "Buy signal threshold",
    0.50, 0.80, 0.55, 0.01
)

st.markdown("---")
st.subheader("🖼️ চার্টের ছবি দিয়ে বিশ্লেষণ")

uploaded_chart = st.file_uploader(
    "চার্টের স্ক্রিনশট আপলোড করুন",
    type=["png", "jpg", "jpeg"],
    help="TradingView, Google Finance বা Broker অ্যাপের চার্ট ছবি দিন"
)

image_weight = st.slider(
    "ছবির গুরুত্ব",
    0.0, 0.5, 0.15, 0.05
)

run_btn = st.button("চালান", type="primary", use_container_width=True)


# ============================================================
# Main logic
# ============================================================
if run_btn:
    try:
        start = validate_start_date(start)

        with st.spinner(f"{ticker} এর ডেটা আসছে এবং analysis চলছে..."):
            model, df, split_idx, X_test, y_test = train_model_cached(
                ticker=ticker,
                start=start,
                test_ratio=test_ratio,
                n_estimators=n_estimators
            )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        bt = backtest(df, y_pred, y_prob, split_idx, threshold)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        latest = df[FEATURES].iloc[[-1]].values
        ml_probability = float(model.predict_proba(latest)[0, 1])

        smc = analyze_smart_money(df)
        wyckoff = analyze_wyckoff(df)
        gann = analyze_gann(df)
        elliott = analyze_elliott_wave(df)

        image_result = None

        if uploaded_chart is not None:
            image_result = analyze_chart_image(uploaded_chart)

        combined_probability, final_signal = combine_all_predictions(
            ml_probability=ml_probability,
            smc_score=smc["score"],
            wyckoff_score=wyckoff["score"],
            gann_score=gann["score"],
            elliott_score=elliott["score"],
            image_score=image_result["score"] if image_result else None,
            image_weight=image_weight if image_result else 0.0
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc * 100:.1f}%")
        m2.metric("Precision", f"{prec * 100:.1f}%")
        m3.metric("Recall", f"{rec * 100:.1f}%")
        m4.metric("Combined probability", f"{combined_probability * 100:.1f}%")

        st.success(f"**চূড়ান্ত signal:** {final_signal}")

        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
            [
                "📊 দাম ও এভারেজ",
                "📉 RSI",
                "📈 MACD",
                "💰 Backtest",
                "🧠 Advanced Analysis",
                "📄 ডেটা",
                "🖼️ ছবি বিশ্লেষণ"
            ]
        )

        with tab1:
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(df.index, df["Close"], label="Close", color="black")
            ax.plot(df.index, df["sma20"], label="SMA 20", color="blue", alpha=0.7)
            ax.plot(df.index, df["sma50"], label="SMA 50", color="orange", alpha=0.7)
            ax.axvline(df.index[split_idx], color="red", linestyle="--", label="Train/Test split")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_title(f"{ticker} — দাম ও মুভিং এভারেজ")
            st.pyplot(fig)
            plt.close(fig)

        with tab2:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df.index, df["rsi14"], color="purple")
            ax.axhline(70, color="red", linestyle="--", label="Overbought 70")
            ax.axhline(30, color="green", linestyle="--", label="Oversold 30")
            ax.axvline(df.index[split_idx], color="red", linestyle="--")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_title("RSI 14")
            st.pyplot(fig)
            plt.close(fig)

        with tab3:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(df.index, df["macd"], label="MACD", color="blue")
            ax.plot(df.index, df["macd_signal"], label="Signal", color="orange")
            ax.bar(df.index, df["macd_hist"], label="Histogram", color="gray", alpha=0.4)
            ax.axvline(df.index[split_idx], color="red", linestyle="--")
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_title("MACD (12, 26, 9)")
            st.pyplot(fig)
            plt.close(fig)

        with tab4:
            tdf = bt["test_df"]

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(tdf.index, tdf["cum_strategy"], label="Strategy", color="green", linewidth=2)
            ax.plot(tdf.index, tdf["cum_buyhold"], label="Buy & Hold", color="gray", linewidth=2)
            ax.legend()
            ax.grid(alpha=0.3)
            ax.set_title("Backtest (transaction cost সহ)")
            st.pyplot(fig)
            plt.close(fig)

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Strategy return", f'{bt["strategy_return"] * 100:.2f}%')
            b2.metric("Buy & Hold return", f'{bt["buyhold_return"] * 100:.2f}%')
            b3.metric("Max drawdown", f'{bt["max_drawdown"] * 100:.2f}%')
            b4.metric("Sharpe ratio", f'{bt["sharpe"]:.2f}')

            c1, c2, c3 = st.columns(3)
            c1.metric("মোট trade", bt["trade_count"])
            c2.metric("Win rate", f'{bt["win_rate"] * 100:.1f}%')
            c3.metric("Threshold", f"{threshold:.2f}")

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            st.dataframe(pd.DataFrame(
                cm,
                index=["আসল 0", "আসল 1"],
                columns=["পূর্বাভাস 0", "পূর্বাভাস 1"]
            ))

        with tab5:
            st.subheader("🧠 Smart Money Concepts")
            st.metric("SMC signal", smc["label"])
            st.metric("SMC score", f"{smc['score']:.2f}")

            for note in smc["notes"]:
                st.write(f"- {note}")

            st.markdown("---")
            st.subheader("📊 Wyckoff Analysis")
            st.metric("Wyckoff signal", wyckoff["label"])
            st.metric("Wyckoff score", f"{wyckoff['score']:.2f}")

            for note in wyckoff["notes"]:
                st.write(f"- {note}")

            st.markdown("---")
            st.subheader("🔢 Gann Square of 9")
            st.metric("Gann signal", gann["label"])
            st.metric("বর্তমান দাম", f"{gann['price']:.2f}")
            st.metric("পরবর্তী support", f"{gann['next_support']:.2f}")
            st.metric("পরবর্তী resistance", f"{gann['next_resistance']:.2f}")

            st.markdown("---")
            st.subheader("🌊 Elliott Wave Analysis")
            st.metric("Elliott signal", elliott["label"])
            st.metric("Elliott score", f"{elliott['score']:.2f}")

            for note in elliott["notes"]:
                st.write(f"- {note}")

            st.info(
                "এই analysis গুলো rule-based approximation। "
                "বাস্তব SMC, Wyckoff, Gann ও Elliott Wave analysis-এ মানুষের subjective chart reading জড়িত।"
            )

        with tab6:
            display_df = df[[
                "Close", "ret_1d", "sma20", "sma50",
                "rsi14", "macd", "macd_signal", "target"
            ]].tail(100)

            st.dataframe(display_df, use_container_width=True)

            csv_buffer = io.StringIO()
            display_df.to_csv(csv_buffer, index=True)

            st.download_button(
                label="শেষ ১০০ দিনের ডেটা CSV ডাউনলোড করুন",
                data=csv_buffer.getvalue(),
                file_name=f"{ticker}_recent_data.csv",
                mime="text/csv",
                use_container_width=True
            )

        with tab7:
            if image_result is None:
                st.warning("ছবি বিশ্লেষণ দেখতে উপরে একটি চার্টের ছবি আপলোড করুন।")
            else:
                i1, i2, i3 = st.columns(3)
                i1.metric("ছবির trend", image_result["label"])
                i2.metric("শনাক্ত line segment", image_result["detected_lines"])
                i3.metric("Combined probability", f"{combined_probability * 100:.1f}%")

                st.metric("চূড়ান্ত signal", final_signal)

                st.markdown("#### Edge-detected chart")
                st.image(
                    image_result["edges"],
                    caption="OpenCV দিয়ে শনাক্ত করা edge",
                    use_container_width=True
                )

        st.markdown("---")
        st.subheader("ফিচারের গুরুত্ব")

        importance = pd.Series(
            model.feature_importances_,
            index=FEATURES
        ).sort_values(ascending=False)

        st.bar_chart(importance)

        st.caption(
            "⚠️ এটি শিক্ষামূলক উদাহরণ। SMC, Wyckoff, Gann ও Elliott Wave কোনো নিশ্চিত prediction দেয় না। "
            "বিনিয়োগ সিদ্ধান্তের আগে আর্থিক পরামর্শ নিন।"
        )

    except Exception as e:
        st.error(f"ত্রুটি: {e}")
        st.info("Ticker symbol, তারিখ, ছবি এবং ইন্টারনেট সংযোগ দেখুন। যেমন: AAPL, TSLA, RELIANCE.NS")
