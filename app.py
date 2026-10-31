
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# ============================================================
# Settings
# ============================================================
FEATURES = [
    "ret_1d", "sma20", "sma50", "ema12", "rsi14",
    "macd", "macd_signal", "macd_hist", "vol_change",
    "sma20_gap", "sma50_gap"
]

TRANSACTION_COST = 0.002

# set_page_config অবশ্যই প্রথম Streamlit command হতে হবে
st.set_page_config(
    page_title="Share Market AI",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# Data functions
# ============================================================
def download_price_data(ticker, start="2015-01-01", end=None):
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(f"{ticker}: কোনো ডেটা পাওয়া যায়নি। Ticker symbol দেখুন।")

    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def add_indicators(df):
    close = df["Close"]
    volume = df["Volume"]

    df["ret_1d"] = close.pct_change(1)
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
    df["sma20_gap"] = (close - df["sma20"]) / df["sma20"]
    df["sma50_gap"] = (close - df["sma50"]) / df["sma50"]

    return df


def add_target(df, horizon=1):
    df["future_return"] = df["Close"].pct_change(horizon).shift(-horizon)
    df["target"] = (df["future_return"] > 0).astype(int)
    df = df.dropna()

    if df.empty:
        raise ValueError("পর্যাপ্ত ডেটা নেই। শুরুর তারিখ আরও পুরনো করুন।")

    return df


def make_model(n_estimators=300):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )


# ============================================================
# Cached functions
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def load_prepared_data(ticker, start):
    """ডেটা নামিয়ে indicator ও target তৈরি করে; ফলাফল ১ ঘণ্টা cache রাখে।"""
    df = download_price_data(ticker, start=start)
    df = add_indicators(df)
    df = add_target(df)
    return df


@st.cache_resource(show_spinner=False, max_entries=10)
def train_model_cached(ticker, start, test_ratio, n_estimators=300):
    """মডেল train করে এবং model object cache রাখে।"""
    df = load_prepared_data(ticker, start)

    X = df[FEATURES].values
    y = df["target"].values

    split_idx = int(len(X) * (1 - test_ratio))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = make_model(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    return model, df, split_idx, X_test, y_test


# ============================================================
# Backtest
# ============================================================
def backtest(df, y_pred, split_idx, cost=TRANSACTION_COST):
    test_df = df.iloc[split_idx:].copy()
    test_df["signal"] = y_pred
    test_df["position"] = test_df["signal"].shift(1).fillna(0)

    test_df["strategy_return"] = test_df["position"] * test_df["ret_1d"]

    position_change = test_df["position"].diff().abs().fillna(0)
    test_df["net_return"] = test_df["strategy_return"] - (position_change * cost)

    test_df["cum_strategy"] = (1 + test_df["net_return"]).cumprod()
    test_df["cum_buyhold"] = (1 + test_df["ret_1d"]).cumprod()

    return {
        "test_df": test_df,
        "strategy_return": float(test_df["cum_strategy"].iloc[-1] - 1),
        "buyhold_return": float(test_df["cum_buyhold"].iloc[-1] - 1),
        "trade_count": int(test_df["position"].diff().abs().sum())
    }


# ============================================================
# UI
# ============================================================
st.title("📈 শেয়ার মার্কেট Chart Analysis ও Prediction AI")
st.caption("শিক্ষামূলক উদাহরণ — guaranteed profit নয়।")

# মোবাইল থেকে সরাসরি টিকার লেখা যাবে
popular = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "NVDA", "META", "RELIANCE.NS", "TCS.NS"
]

ticker = st.selectbox(
    "জনপ্রিয় টিকার বেছে নিন",
    options=popular,
    index=0
)

custom_ticker = st.text_input(
    "অথবা নিজের ticker symbol লিখুন",
    placeholder="যেমন: AAPL, TSLA, RELIANCE.NS",
    help="মোবাইল থেকেও এখানে লিখে সার্চ করতে পারবেন"
).upper().strip()

if custom_ticker:
    ticker = custom_ticker

start = st.text_input("শুরুর তারিখ", value="2015-01-01")
test_ratio = st.slider("Test data ratio", 0.1, 0.4, 0.2, 0.05)
n_estimators = st.slider("Model trees", 100, 500, 150, 50)

run_btn = st.button("চালান", type="primary", use_container_width=True)


# ============================================================
# Main logic
# ============================================================
if run_btn:
    try:
        with st.spinner(f"{ticker} এর ডেটা আসছে ও মডেল train হচ্ছে..."):
            model, df, split_idx, X_test, y_test = train_model_cached(
                ticker=ticker,
                start=start,
                test_ratio=test_ratio,
                n_estimators=n_estimators
            )

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        bt = backtest(df, y_pred, split_idx)

        latest = df[FEATURES].iloc[[-1]].values
        latest_prob = model.predict_proba(latest)[0, 1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{acc * 100:.1f}%")
        c2.metric("Precision", f"{prec * 100:.1f}%")
        c3.metric("Recall", f"{rec * 100:.1f}%")
        c4.metric("পরের দিন বাড়ার সম্ভাবনা", f"{latest_prob * 100:.1f}%")

        st.markdown("---")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📊 দাম ও এভারেজ", "📉 RSI", "📈 MACD", "💰 Backtest"]
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

            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Strategy return", f'{bt["strategy_return"] * 100:.2f}%')
            cc2.metric("Buy & Hold return", f'{bt["buyhold_return"] * 100:.2f}%')
            cc3.metric("মোট trade", bt["trade_count"])

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)

            st.dataframe(pd.DataFrame(
                cm,
                index=["আসল 0", "আসল 1"],
                columns=["পূর্বাভাস 0", "পূর্বাভাস 1"]
            ))

        st.markdown("---")
        st.subheader("ফিচারের গুরুত্ব")

        importance = pd.Series(
            model.feature_importances_,
            index=FEATURES
        ).sort_values(ascending=False)

        st.bar_chart(importance)

        st.caption("⚠️ এটি শিক্ষামূলক উদাহরণ। বিনিয়োগ সিদ্ধান্তের আগে আর্থিক পরামর্শ নিন।")

    except Exception as e:
        st.error(f"ত্রুটি: {e}")
        st.info("Ticker symbol ও ইন্টারনেট সংযোগ দেখুন। যেমন: AAPL, TSLA, RELIANCE.NS")
