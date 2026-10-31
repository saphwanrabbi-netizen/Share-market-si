import io
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
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
# Data functions
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
    n_estimators: int,
    threshold: float
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
# UI
# ============================================================
st.title("📈 শেয়ার মার্কেট Chart Analysis ও Prediction AI")
st.caption("শিক্ষামূলক উদাহরণ — guaranteed profit নয়।")

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
        help="মোবাইল থেকেও এখানে লিখতে পারবেন"
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
    "Buy signal probability threshold",
    0.50, 0.80, 0.55, 0.01,
    help="যত বেশি threshold, তত কম কিন্তু তুলনামূলক নিশ্চিত signal"
)

run_btn = st.button("চালান", type="primary", use_container_width=True)


# ============================================================
# Main logic
# ============================================================
if run_btn:
    try:
        start = validate_start_date(start)

        with st.spinner(f"{ticker} এর ডেটা আসছে ও মডেল train হচ্ছে..."):
            model, df, split_idx, X_test, y_test = train_model_cached(
                ticker=ticker,
                start=start,
                test_ratio=test_ratio,
                n_estimators=n_estimators,
                threshold=threshold
            )

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        bt = backtest(df, y_pred, y_prob, split_idx, threshold)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        latest = df[FEATURES].iloc[[-1]].values
        latest_prob = float(model.predict_proba(latest)[0, 1])
        latest_signal = "কিনুন" if latest_prob >= threshold else "অপেক্ষা করুন"

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{acc * 100:.1f}%")
        m2.metric("Precision", f"{prec * 100:.1f}%")
        m3.metric("Recall", f"{rec * 100:.1f}%")
        m4.metric("পরের দিন বাড়ার সম্ভাবনা", f"{latest_prob * 100:.1f}%")

        st.info(f"বর্তমান মডেল সিগন্যাল: **{latest_signal}**")

        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📊 দাম ও এভারেজ", "📉 RSI", "📈 MACD", "💰 Backtest", "📄 ডেটা"]
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

        st.markdown("---")
        st.subheader("ফিচারের গুরুত্ব")

        importance = pd.Series(
            model.feature_importances_,
            index=FEATURES
        ).sort_values(ascending=False)

        st.bar_chart(importance)

        st.caption(
            "⚠️ এটি শিক্ষামূলক উদাহরণ। অতীতের backtest result ভবিষ্যতের রিটার্নের নিশ্চয়তা দেয় না। "
            "বিনিয়োগ সিদ্ধান্তের আগে আর্থিক পরামর্শ নিন।"
        )

    except Exception as e:
        st.error(f"ত্রুটি: {e}")
        st.info("Ticker symbol, তারিখ এবং ইন্টারনেট সংযোগ দেখুন। যেমন: AAPL,BINANCE TSLA, RELIANCE.NS")
