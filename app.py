# ============================================================
# app.py — শেয়ার মার্কেট Chart Analysis ও Prediction AI
# Streamlit Dashboard + সব prediction logic একটিমাত্র ফাইলে
#
# চালানোর কমান্ড:  streamlit run app.py
# ============================================================

# --- দরকারি লাইব্রেরি ---
# yfinance  = Yahoo Finance থেকে historical দাম আনা
# pandas    = ডেটা টেবিল নিয়ে কাজ করা
# numpy     = সংখ্যা নিয়ে হিসাব (এখানে ব্যবহার কম, তবে রাখলাম)
# matplotlib= চার্ট আঁকা
# sklearn   = Random Forest মডেল ও evaluation metrics
# streamlit = dashboard UI
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # Streamlit-এ চার্ট দেখাতে এই backend দরকার
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, confusion_matrix)

# ============================================================
# অংশ ১: সব feature-এর নামের list (১১টি)
# প্রতিটিই শুধু অতীতের দাম থেকে তৈরি → look-ahead bias নেই
# ============================================================
FEATURES = [
    "ret_1d",        # দৈনিক return
    "sma20",         # সিম্পল মুভিং এভারেজ ২০
    "sma50",         # সিম্পল মুভিং এভারেজ ৫০
    "ema12",         # এক্সপোনেনশিয়াল মুভিং এভারেজ ১২
    "rsi14",         # RSI 14
    "macd",          # MACD লাইন
    "macd_signal",   # MACD signal লাইন
    "macd_hist",     # MACD histogram
    "vol_change",    # volume পরিবর্তন
    "sma20_gap",     # দাম SMA20 থেকে কত % দূরে
    "sma50_gap",     # দাম SMA50 থেকে কত % দূরে
]

TRANSACTION_COST = 0.002   # প্রতি trade-এ খরচ (০.২%)


# ============================================================
# অংশ ২: download_price_data — Yahoo থেকে ডেটা নামায়
# MultiIndex column-এর সমস্যা এখানেই ঠিক করা হয়
# ============================================================
def download_price_data(ticker, start="2015-01-01", end=None):
    """
    একটি ticker-এর historical দাম নামায় এবং পরিষ্কার করে।

    ticker : শেয়ারের symbol (যেমন "AAPL", "RELIANCE.NS")
    start  : কত তারিখ থেকে ডেটা (string, "2015-01-01")
    end    : কত তারিখ পর্যন্ত (None = আজ পর্যন্ত)

    ফেরত দেয়: pandas DataFrame (Open, High, Low, Close, Volume)
    """
    # yfinance দিয়ে ডেটা ডাউনলোড
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,   # split/dividend adjust করা দাম
        progress=False      # terminal-এ progress bar দেখাব না
    )

    # নতুন yfinance version-এ MultiIndex column দেয়, যেমন ('Close', 'AAPL')
    # তখন df["Close"] কাজ করে না — তাই আগে এক-স্তরে নামাতে হয়
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # কোনো ডেটা না পাওয়া গেলে স্পষ্ট error দিচ্ছি
    if df.empty:
        raise ValueError(
            f"{ticker}: কোনো ডেটা পাওয়া যায়নি। "
            "Ticker symbol ঠিক আছে কিনা দেখুন।"
        )

    # শুধু দরকারি ৫টি column রাখছি, NaN সারি বাদ দিচ্ছি
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

    # ticker-এর নাম ধরে রাখছি
    df.attrs["ticker"] = ticker

    return df


# ============================================================
# অংশ ৩: add_indicators — সব technical indicator তৈরি
# প্রতিটি indicator শুধু অতীতের দাম ব্যবহার করে
# ============================================================
def add_indicators(df):
    """
    df-এ ret_1d, sma20, sma50, ema12, rsi14, macd, macd_signal,
    macd_hist, vol_change, sma20_gap, sma50_gap যোগ করে।

    df : download_price_data থেকে পাওয়া DataFrame
    ফেরত দেয়: একই DataFrame, নতুন column সহ
    """
    close = df["Close"]     # বন্ধ দাম
    volume = df["Volume"]   # volume

    # --- Daily Return: আজকের দাম গতকালের তুলনায় কত % বদলাল ---
    df["ret_1d"] = close.pct_change(1)

    # --- SMA 20 ও SMA 50: শেষ ২০ ও ৫০ দিনের গড় ---
    df["sma20"] = close.rolling(window=20).mean()
    df["sma50"] = close.rolling(window=50).mean()

    # --- EMA 12: সাম্প্রতিক দামকে বেশি গুরুত্ব দেয় ---
    df["ema12"] = close.ewm(span=12, adjust=False).mean()

    # --- RSI 14 ---
    # RSI ৭০+ = overbought, ৩০- = oversold
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)     # শূন্য দিয়ে ভাগ আটকাতে ছোট মান যোগ
    df["rsi14"] = 100 - (100 / (1 + rs))

    # --- MACD (12, 26, 9) ---
    ema12_fast = close.ewm(span=12, adjust=False).mean()
    ema26_slow = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12_fast - ema26_slow
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # --- Volume Change: আজকের volume গতকালের তুলনায় কত % বদলাল ---
    df["vol_change"] = volume.pct_change(1)

    # --- SMA থেকে দাম কত % দূরে ---
    df["sma20_gap"] = (close - df["sma20"]) / df["sma20"]
    df["sma50_gap"] = (close - df["sma50"]) / df["sma50"]

    return df


# ============================================================
# অংশ ৪: add_target — পরের দিন বাড়লে 1, নাহলে 0
# ============================================================
def add_target(df, horizon=1):
    """
    আগামী horizon দিন পরের return দেখে target বানায়।
    horizon=1 মানে পরের দিন।

    বিঃদ্রঃ shift(-horizon) ইচ্ছাকৃতভাবে future দিকে যায় (এটা উত্তর/y),
    কিন্তু ফিচার (X) কখনো future ব্যবহার করে না — তাই লিক নেই।

    ফেরত দেয়: পরিষ্কার DataFrame (NaN বাদ দেওয়ার পর)
    """
    # future_return = horizon দিন পরের return
    df["future_return"] = df["Close"].pct_change(horizon).shift(-horizon)

    # দাম বাড়লে 1, কমলে বা বরাবর হলে 0
    df["target"] = (df["future_return"] > 0).astype(int)

    # rolling window-এর শুরুতে ও শেষে NaN থাকে — বাদ দিচ্ছি
    df = df.dropna()

    if df.empty:
        raise ValueError(
            "Target বানানোর পর কোনো ডেটা থাকল না। "
            "start তারিখ আরও পুরনো করুন (যেমন 2012-01-01)।"
        )

    return df


# ============================================================
# অংশ ৫: make_model — Random Forest তৈরি (শেখানো হয়নি)
# ============================================================
def make_model():
    """
    RandomForestClassifier return করে।
    n_estimators=300, max_depth=6, min_samples_leaf=20,
    random_state=42, n_jobs=-1
    """
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1,
    )


# ============================================================
# অংশ ৬: backtest — strategy বনাম buy & hold
# ============================================================
def backtest(df, y_pred, split_idx, cost=TRANSACTION_COST):
    """
    সহজ backtest: signal=1 হলে কিনুন, পরের দিন বিক্রি করুন।

    ফেরত দেয় dict:
      test_df          = test period-এর DataFrame
                         (cum_strategy ও cum_buyhold column সহ)
      strategy_return  = strategy-র মোট return
      buyhold_return   = buy & hold-এর মোট return
      trade_count      = মোট trade সংখ্যা
    """
    # test period-এর অংশটুকু আলাদা করছি
    test_df = df.iloc[split_idx:].copy()
    test_df["signal"] = y_pred

    # ------------------------------------------------------------
    # Look-ahead bias প্রতিরোধ:
    # গতকালের signal দিয়েই আজকের position ঠিক করছি (shift(1))।
    # ------------------------------------------------------------
    test_df["position"] = test_df["signal"].shift(1).fillna(0)

    # position অনুযায়ী আজকের return
    test_df["strategy_return"] = test_df["position"] * test_df["ret_1d"]

    # position বদলালেই transaction cost কাটছি
    position_change = test_df["position"].diff().abs().fillna(0)
    test_df["cost"] = position_change * cost
    test_df["net_return"] = test_df["strategy_return"] - test_df["cost"]

    # ক্রমবর্ধমান return
    test_df["cum_strategy"] = (1 + test_df["net_return"]).cumprod()
    test_df["cum_buyhold"] = (1 + test_df["ret_1d"]).cumprod()

    strategy_return = float(test_df["cum_strategy"].iloc[-1] - 1)
    buyhold_return = float(test_df["cum_buyhold"].iloc[-1] - 1)
    trade_count = int(test_df["position"].diff().abs().sum())

    return {
        "test_df": test_df,
        "strategy_return": strategy_return,
        "buyhold_return": buyhold_return,
        "trade_count": trade_count,
    }


# ============================================================
# অংশ ৭: Streamlit UI শুরু
# ============================================================
st.set_page_config(page_title="Share Market AI", layout="wide")

st.title("📈 শেয়ার মার্কেট Chart Analysis ও Prediction AI")
st.caption("শিক্ষামূলক উদাহরণ — guaranteed profit নয়।")

# ---- sidebar-এ ইনপুট ----
with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker symbol", value="AAPL").upper()
    start = st.text_input("শুরুর তারিখ", value="2015-01-01")
    test_ratio = st.slider("Test data ratio", 0.1, 0.4, 0.2, 0.05)
    st.markdown("---")
    st.markdown("**উদাহরণ:** `AAPL`, `TSLA`, `RELIANCE.NS`, `BTC-USD`, `^GSPC`")
    run_btn = st.button("চালান", type="primary")


try:
    # ডেটা নামানো ও প্রস্তুত করা
    with st.spinner(f"{ticker} এর ডেটা আসছে..."):
        df = download_price_data(ticker, start=start)
        df = add_indicators(df)
        df = add_target(df)

    X = df[FEATURES].values
    y = df["target"].values

    # time-series split — random shuffle নেই (look-ahead bias আটকায়)
    split_idx = int(len(X) * (1 - test_ratio))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # মডেল শেখানো
    model = make_model()
    model.fit(X_train, y_train)

    # পূর্বাভাস
    y_pred = model.predict(X_test)

    # মেট্রিক্স
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    # backtest
    bt = backtest(df, y_pred, split_idx)

    # সবশেষ দিনের prediction
    latest = df[FEATURES].iloc[[-1]].values
    latest_prob = model.predict_proba(latest)[0, 1]

    # ================= KPI =================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc * 100:.1f}%")
    c2.metric("Precision", f"{prec * 100:.1f}%")
    c3.metric("Recall", f"{rec * 100:.1f}%")
    c4.metric("পরের দিন বাড়ার সম্ভাবনা", f"{latest_prob * 100:.1f}%")

    st.markdown("---")

    # ================= Charts =================
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 দাম ও এভারেজ", "📉 RSI", "📈 MACD", "💰 Backtest"]
    )

    # --- Tab 1: দাম ও SMA ---
    with tab1:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df.index, df["Close"], label="Close", color="black")
        ax.plot(df.index, df["sma20"], label="SMA 20", color="blue", alpha=0.7)
        ax.plot(df.index, df["sma50"], label="SMA 50", color="orange", alpha=0.7)
        ax.axvline(df.index[split_idx], color="red",
                   linestyle="--", label="Train/Test split")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title(f"{ticker} — দাম ও মুভিং এভারেজ")
        st.pyplot(fig)
        plt.close(fig)     # memory ছাড়তে

    # --- Tab 2: RSI ---
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

    # --- Tab 3: MACD ---
    with tab3:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df["macd"], label="MACD", color="blue")
        ax.plot(df.index, df["macd_signal"], label="Signal", color="orange")
        ax.bar(df.index, df["macd_hist"], label="Histogram",
               color="gray", alpha=0.4)
        ax.axvline(df.index[split_idx], color="red", linestyle="--")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title("MACD (12, 26, 9)")
        st.pyplot(fig)
        plt.close(fig)

    # --- Tab 4: Backtest ---
    with tab4:
        tdf = bt["test_df"]
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(tdf.index, tdf["cum_strategy"],
                label="Strategy", color="green", linewidth=2)
        ax.plot(tdf.index, tdf["cum_buyhold"],
                label="Buy & Hold", color="gray", linewidth=2)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title("Backtest (transaction cost সহ)")
        st.pyplot(fig)
        plt.close(fig)

        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Strategy return",
                   f'{bt["strategy_return"] * 100:.2f}%')
        cc2.metric("Buy & Hold return",
                   f'{bt["buyhold_return"] * 100:.2f}%')
        cc3.metric("মোট trade", bt["trade_count"])

        st.subheader("Confusion Matrix")
        st.write("row = আসল, col = পূর্বাভাস")
        cm = confusion_matrix(y_test, y_pred)
        st.dataframe(pd.DataFrame(
            cm,
            index=["আসল 0", "আসল 1"],
            columns=["পূর্বাভাস 0", "পূর্বাভাস 1"]
        ))

    # ---- feature importance ----
    st.markdown("---")
    st.subheader("ফিচারের গুরুত্ব")
    importance = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    st.bar_chart(importance)

    st.caption(
        "⚠️ এটি শিক্ষামূলক উদাহরণ। বিনিয়োগ সিদ্ধান্তের আগে "
        "একজন আর্থিক পরামর্শকের সাথে কথা বলুন।"
    )

except Exception as e:
    # কোনো সমস্যা হলে dashboard-এ স্পষ্ট বার্তা দেখাব
    st.error(f"ত্রুটি: {e}")
    st.info(
        "Ticker symbol ঠিক আছে কিনা দেখুন "
        "(যেমন AAPL, TSLA, RELIANCE.NS)। "
        "ইন্টারনেট সংযোগও চেক করুন।"
    )
