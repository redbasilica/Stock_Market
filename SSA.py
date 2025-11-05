import streamlit as st
import yfinance as yf
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from data import scrape_stock_data, scrape_kap_data, update_dataframe_types

st.set_page_config(page_title="BIST Hisse Analiz", layout="centered")

st.title("📈 Hisse Analiz")

# ---------------------- Ana ekranda kullanım ----------------------

st.markdown("""
## Kullanım

1. **Filtre Ayarları** panelinden teknik filtrelerin parametrelerini seçin:  
   - MA yakınlık toleransı  
   - Hacim artış eşiği  
   - RSI dip seviyesi (isteğe bağlı)  
   - Bugün tavan yapan hisseleri filtreleme

2. Tarama yapmak istediğiniz hisseleri seçin veya boş bırakarak tüm hisseleri tarayın.

3. **Taramayı Başlat** butonuna tıklayın.

4. Filtreleme sonuçları listelenecek, her hisse için detaylı bilgiler ve teknik grafikler gösterilecektir.
""")


# ---------------------- Veri yükleme ----------------------

@st.cache_data
def load_market_data():
    """
    Load market data using data.py functions
    Returns df_ozet dataframe and derived dictionaries
    """
    # Scrape data from İş Yatırım
    df_ozet = scrape_stock_data()

    if df_ozet is None:
        st.error("İş Yatırım verisi alınamadı!")
        return None, [], {}, {}

    # Scrape data from KAP
    df_temp = scrape_kap_data()

    if df_temp is not None:
        # Update df_ozet with KAP data
        df_ozet = update_dataframe_types(df_ozet, df_temp)

    # Get list of tickers from df_ozet
    tickers = []
    if 'Kod' in df_ozet.columns:
        # Clean and prepare tickers
        df_ozet['Kod'] = df_ozet['Kod'].str.strip().str.upper()
        tickers = [f"{ticker}.IS" for ticker in df_ozet['Kod'].tolist() if pd.notna(ticker) and ticker != '']

    # Create halka açıklık dictionary
    halka_aciklik_dict = {}
    if 'Halka AçıklıkOranı (%)' in df_ozet.columns:
        for idx, row in df_ozet.iterrows():
            kod = row.get('Kod', '')
            halka_aciklik = row.get('Halka AçıklıkOranı (%)', np.nan)
            if kod and not pd.isna(halka_aciklik):
                halka_aciklik_dict[kod] = halka_aciklik * 100  # Convert back to percentage for display

    # Create dolaşımdaki lot dictionary
    dolasim_lot_dict = {}
    if 'Fiili Dolaşımdaki Pay Tutarı(TL)' in df_ozet.columns and 'Kapanış(TL)' in df_ozet.columns:
        for idx, row in df_ozet.iterrows():
            kod = row.get('Kod', '')
            fiili_dolasim = row.get('Fiili Dolaşımdaki Pay Tutarı(TL)', np.nan)
            kapanis = row.get('Kapanış(TL)', np.nan)

            if kod and not pd.isna(fiili_dolasim) and not pd.isna(kapanis) and kapanis > 0:
                # Calculate lot (1 lot = 100 shares)
                dolasim_lot = fiili_dolasim / (kapanis * 100)
                dolasim_lot_dict[kod] = int(dolasim_lot)

    return df_ozet, tickers, halka_aciklik_dict, dolasim_lot_dict


# Load data
with st.spinner("Piyasa verileri yükleniyor..."):
    df_ozet, all_tickers, halka_aciklik_dict, dolasim_lot_dict = load_market_data()

if df_ozet is None or not all_tickers:
    st.error("Veri yüklenemedi. Lütfen internet bağlantınızı kontrol edin ve sayfayı yenileyin.")
    st.stop()


# ---------------------- Teknik hesaplamalar ----------------------

def calculate_rsi(series, period=14):
    """Thread-safe RSI calculation"""
    with threading.Lock():
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Thread-safe MACD calculation"""
    with threading.Lock():
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram


# ---------------------- Grafik hazırlama ----------------------

def prepare_data_for_plot(ticker):
    """Thread-safe data preparation for plotting"""
    try:
        data = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
        if data.empty or len(data) < 50:
            return None

        data = data.copy()  # Create a copy to avoid shared data issues
        data.dropna(inplace=True)

        # Calculate indicators in thread-safe manner
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["MA200"] = data["Close"].rolling(200).mean()
        data["EMA89"] = data["Close"].ewm(span=89, adjust=False).mean()
        data["RSI"] = calculate_rsi(data["Close"])
        macd_line, signal_line, histogram = calculate_macd(data["Close"])
        data["MACD_Line"] = macd_line
        data["MACD_Signal"] = signal_line
        data["MACD_Hist"] = histogram
        return data
    except Exception as e:
        return None


def plot_stock_chart(data, ticker_name):
    """Thread-safe chart plotting"""
    with threading.Lock():  # Ensure matplotlib operations are thread-safe
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                            gridspec_kw={'height_ratios': [2, 1, 1]})

        ax1.plot(data.index, data["Close"], label="Kapanış", color="blue")
        ax1.plot(data.index, data["MA20"], label="MA20", color="orange")
        ax1.plot(data.index, data["MA50"], label="MA50", color="green")
        ax1.plot(data.index, data["MA200"], label="MA200", color="red")
        ax1.plot(data.index, data["EMA89"], label="EMA89", color="magenta", linestyle="--")
        ax1.set_title(f"{ticker_name} - Son 1 Yıl Teknik Görünüm")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(data.index, data["RSI"], label="RSI", color="purple")
        ax2.axhline(70, color='red', linestyle='--', linewidth=1)
        ax2.axhline(30, color='green', linestyle='--', linewidth=1)
        ax2.set_ylabel("RSI")
        ax2.legend()
        ax2.grid(True)

        ax3.plot(data.index, data["MACD_Line"], label="MACD", color="blue")
        ax3.plot(data.index, data["MACD_Signal"], label="Signal", color="orange")
        ax3.bar(data.index, data["MACD_Hist"], label="Histogram", color="gray", alpha=0.4)
        ax3.set_ylabel("MACD")
        ax3.legend()
        ax3.grid(True)

        # İmza ekleme
        fig.text(0.5, 0.5, 'Bay-P',
                 fontsize=50, color='gray', alpha=0.15,
                 ha='center', va='center',
                 weight='bold', style='italic', rotation=20)

        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()


# ---------------------- Tarama fonksiyonu ----------------------

def scan_single_stock(ticker, ma_tolerance, volume_threshold, use_ma, use_volume, use_rsi, rsi_threshold,
                      ceiling_threshold):
    """Scan a single stock - designed for parallel execution"""
    try:
        data = yf.download(ticker, period="90d", interval="1d", progress=False, threads=False)
        if data.empty or len(data) < 30:
            return None

        data = data.copy()  # Create a copy to avoid shared data issues
        data.dropna(inplace=True)

        # Calculate indicators
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["MA200"] = data["Close"].rolling(200).mean()
        data["AvgVolume20"] = data["Volume"].rolling(20).mean()
        data["RSI"] = calculate_rsi(data["Close"])

        close = float(data["Close"].iloc[-1])
        prev_close = float(data["Close"].iloc[-2])
        change_pct = ((close - prev_close) / prev_close) * 100

        if ceiling_threshold is not None and change_pct < ceiling_threshold:
            return None

        ma20 = float(data["MA20"].iloc[-1])
        ma50 = float(data["MA50"].iloc[-1])
        ma200 = float(data["MA200"].iloc[-1]) if not pd.isna(data["MA200"].iloc[-1]) else ma50
        rsi_latest = data["RSI"].iloc[-1]
        last_date = data.index[-1].strftime("%Y-%m-%d")
        volume = int(data["Volume"].iloc[-1])
        avg_volume = float(data["AvgVolume20"].iloc[-1])
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0

        is_near_ma = close < min(ma20, ma50, ma200) * (1 + ma_tolerance)
        passes_ma = is_near_ma if use_ma else True
        passes_volume = volume_ratio >= volume_threshold if use_volume else True
        passes_rsi = rsi_latest <= rsi_threshold if use_rsi else True

        if passes_ma and passes_volume and passes_rsi:
            return {
                "Hisse": ticker.replace(".IS", ""),
                "Tarih": last_date,
                "Kapanış": round(close, 2),
                "Değişim": round(change_pct, 2),
                "MA20": round(ma20, 2),
                "MA50": round(ma50, 2),
                "Hacim Katsayısı": round(volume_ratio, 2),
                "RSI": round(rsi_latest, 2)
            }
        return None
    except Exception as e:
        return None


def scan_stocks(tickers, ma_tolerance, volume_threshold, use_ma, use_volume, use_rsi=False, rsi_threshold=30,
                ceiling_threshold=None, max_workers=10):
    """
    Multi-threaded stock scanning for improved performance
    max_workers: number of parallel threads (default 10, adjust based on your system)
    """
    results = []
    total_tickers = len(tickers)
    completed = 0

    # Create a progress bar placeholder
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(
                scan_single_stock,
                ticker,
                ma_tolerance,
                volume_threshold,
                use_ma,
                use_volume,
                use_rsi,
                rsi_threshold,
                ceiling_threshold
            ): ticker for ticker in tickers
        }

        # Process completed tasks
        for future in as_completed(future_to_ticker):
            completed += 1
            ticker = future_to_ticker[future]

            # Update progress
            progress = completed / total_tickers
            progress_bar.progress(progress)
            status_text.text(f"Taranan: {completed}/{total_tickers} - Son: {ticker.replace('.IS', '')}")

            try:
                result = future.result(timeout=30)  # 30 second timeout per stock
                if result is not None:
                    results.append(result)
            except Exception as e:
                # Log error but continue with other stocks
                continue

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    return pd.DataFrame(results)


# ---------------------- Sidebar ----------------------

st.sidebar.header("🔧 Filtre Ayarları")
ma_tolerance = st.sidebar.slider("MA Yakınlık Toleransı (%)", 1, 10, 5) / 100
volume_threshold = st.sidebar.slider("Hacim Artış Eşiği (kat)", 0.0, 5.0, 1.5)
use_ma = st.sidebar.checkbox("MA Dip Filtresi Kullan", value=True)
use_volume = st.sidebar.checkbox("Hacim Filtresi Kullan", value=True)
use_rsi = st.sidebar.checkbox("RSI Dip Filtresi Kullan", value=False)
rsi_threshold = st.sidebar.slider("RSI Eşiği", 10, 50, 30)
use_ceiling_filter = st.sidebar.checkbox("Bugün Tavan Yapanları Tara (≥ %9)", value=False)

# Set default max_workers to 10
max_workers = 10

# Hisse Seçimi - display without .IS suffix for better UX
display_tickers = [t.replace(".IS", "") for t in all_tickers]
selected_display = st.sidebar.multiselect("📌 Tarama İçin Hisse Seç (boş bırak tümü için)", options=display_tickers)
selected_tickers = [f"{t}.IS" for t in selected_display]

# Display data summary
st.sidebar.markdown("---")
st.sidebar.markdown(f"**📊 Veri Özeti**")
st.sidebar.markdown(f"Toplam Hisse: {len(all_tickers)}")
st.sidebar.markdown(f"Halka Açıklık Verisi: {len(halka_aciklik_dict)} hisse")
st.sidebar.markdown(f"Dolaşım Lot Verisi: {len(dolasim_lot_dict)} hisse")


# ---------------------- Ana içerik ----------------------

def fetch_stock_info_parallel(tickers, max_workers=5):
    """Fetch Yahoo Finance info for multiple stocks in parallel"""
    info_dict = {}

    def fetch_single_info(ticker):
        try:
            return ticker, yf.Ticker(ticker).info
        except:
            return ticker, {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_single_info, ticker) for ticker in tickers]
        for future in as_completed(futures):
            ticker, info = future.result()
            info_dict[ticker] = info

    return info_dict


# Create two columns for buttons
col1, col2 = st.columns([1, 1])

with col1:
    scan_button = st.button("🔍 Taramayı Başlat", use_container_width=True)

with col2:
    refresh_button = st.button("🔄 Verileri Yenile", use_container_width=True)

if refresh_button:
    st.cache_data.clear()
    st.rerun()

if scan_button:
    start_time = time.time()

    with st.spinner("Hisseler taranıyor..."):
        tickers_to_scan = selected_tickers if selected_tickers else all_tickers
        ceiling_threshold = 9.5 if use_ceiling_filter else None

        # Multi-threaded scanning
        df = scan_stocks(tickers_to_scan, ma_tolerance, volume_threshold, use_ma, use_volume, use_rsi, rsi_threshold,
                         ceiling_threshold, max_workers=max_workers)

        if df.empty:
            st.warning("Kriterlere uyan hisse bulunamadı.")
        else:
            elapsed_time = time.time() - start_time
            st.success(f"{len(df)} hisse bulundu. (Tarama süresi: {elapsed_time:.1f} saniye)")

            # Fetch all stock info in parallel for found stocks
            found_tickers = [f"{row['Hisse']}.IS" for _, row in df.iterrows()]
            with st.spinner("Hisse bilgileri yükleniyor..."):
                stock_info_dict = fetch_stock_info_parallel(found_tickers, max_workers=5)

            # Try to get USD/TRY rate once
            usdtry = None
            try:
                usdtry = yf.Ticker("USDTRY=X").info.get("regularMarketPrice", None)
            except:
                pass

            # Display results
            for _, row in df.iterrows():
                hisse = row['Hisse']
                ticker_full = hisse + ".IS"
                info = stock_info_dict.get(ticker_full, {})

                market_cap_try = info.get("marketCap", None)
                market_cap_usd_str = "N/A"
                if market_cap_try and usdtry:
                    market_cap_usd = market_cap_try / usdtry
                    if market_cap_usd >= 1e9:
                        market_cap_usd_str = f"{market_cap_usd / 1e9:.2f} Milyar $"
                    elif market_cap_usd >= 1e6:
                        market_cap_usd_str = f"{market_cap_usd / 1e6:.2f} Milyon $"

                # Get data from dictionaries created from df_ozet
                lot = dolasim_lot_dict.get(hisse, "N/A")
                if lot != "N/A":
                    lot = f"{int(lot):,}".replace(",", ".")

                halka_aciklik = halka_aciklik_dict.get(hisse, "N/A")
                if halka_aciklik != "N/A":
                    halka_aciklik = f"%{halka_aciklik:.2f}"

                # Get additional data from df_ozet if available
                piyasa_degeri_tl = "N/A"
                piyasa_degeri_usd = "N/A"
                if 'Kod' in df_ozet.columns:
                    hisse_data = df_ozet[df_ozet['Kod'] == hisse]
                    if not hisse_data.empty:
                        if 'Piyasa Değeri(mn TL)' in df_ozet.columns:
                            pd_tl = hisse_data['Piyasa Değeri(mn TL)'].iloc[0]
                            if not pd.isna(pd_tl):
                                piyasa_degeri_tl = f"{pd_tl:,.0f} mn TL"
                        if 'Piyasa Değeri(mn $)' in df_ozet.columns:
                            pd_usd = hisse_data['Piyasa Değeri(mn $)'].iloc[0]
                            if not pd.isna(pd_usd):
                                piyasa_degeri_usd = f"{pd_usd:,.0f} mn $"

                color = "green" if row['Değişim'] >= 0 else "red"
                sign = "▲" if row['Değişim'] >= 0 else "▼"

                st.markdown(f"""
                <div style="border:1px solid #ccc; border-radius:10px; padding:10px; margin:10px 0;">
                    <strong>{hisse}</strong><br>
                    <i>Tarih: {row['Tarih']}</i><br>
                    Kapanış: <b>{row['Kapanış']}</b> <span style='color:{color}'>{sign} {abs(row['Değişim'])}%</span><br>
                    RSI: <b>{row['RSI']}</b> | Hacim/Ort: <b>{row['Hacim Katsayısı']}</b><br>
                    MA20: {row['MA20']} | MA50: {row['MA50']}<br>
                    <b>Dolaşımdaki Lot:</b> {lot}<br>
                    <b>Halka Açıklık Oranı:</b> {halka_aciklik}<br>
                    <b>Piyasa Değeri (İş Yatırım):</b> {piyasa_degeri_tl} / {piyasa_degeri_usd}<br><br>
                    📊 <b>Finansal Oranlar (Yahoo Finance)</b><br>
                    F/K: <b>{info.get("trailingPE", "N/A")}</b><br>
                    PD/DD: <b>{info.get("priceToBook", "N/A")}</b><br>
                    Piyasa Değeri (YF): <b>{market_cap_usd_str}</b>
                </div>
                """, unsafe_allow_html=True)

                # Prepare and plot chart (these can also be threaded if needed)
                with st.spinner(f"{hisse} grafiği hazırlanıyor..."):
                    data_plot = prepare_data_for_plot(ticker_full)
                    if data_plot is not None:
                        plot_stock_chart(data_plot, hisse)
                    else:
                        st.info(f"{hisse} için yeterli veri bulunamadı.")
