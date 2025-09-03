import streamlit as st
import yfinance as yf
import pandas as pd
import time
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from data_scrap import scrape_stock_data, scrape_circulation_data, merge_dataframes, add_date_column

st.set_page_config(page_title="BIST Hisse Analiz v2", layout="centered")

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


# ---------------------- Veri yükleme (Güncellenmiş) ----------------------

@st.cache_data
def load_scraped_data():
    """Load and merge all scraped data"""
    try:
        # Scrape the first dataset (stock data)
        df_ozet = scrape_stock_data()

        if df_ozet is None:
            st.error("Hisse verileri alınamadı!")
            return None

        # Scrape the second dataset (circulation data)
        df_dolasim = scrape_circulation_data()

        # Merge the dataframes
        df_ozet = merge_dataframes(df_ozet, df_dolasim)

        # Add date column
        df_ozet = add_date_column(df_ozet)

        # Clean the 'Kod' column
        if 'Kod' in df_ozet.columns:
            df_ozet["Kod"] = df_ozet["Kod"].str.strip().str.upper()

        return df_ozet

    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {e}")
        return None


def get_all_bist_tickers():
    """Updated function to get tickers from scraped data"""
    df_ozet = load_scraped_data()

    if df_ozet is None or 'Kod' not in df_ozet.columns:
        st.error("Hisse kodları alınamadı!")
        return []

    # Get unique stock codes and filter out empty/null values
    tickers = df_ozet['Kod'].dropna().unique().tolist()
    tickers = [ticker for ticker in tickers if ticker and str(ticker).strip()]

    return sorted(tickers)


@st.cache_data
def load_halaciklik_data():
    """Updated function to get halka açıklık data from scraped DataFrame"""
    df_ozet = load_scraped_data()

    if df_ozet is None:
        return {}

    # Look for halka açıklık column (case insensitive)
    halka_aciklik_col = None
    for col in df_ozet.columns:
        if 'halka' in col.lower() and 'açık' in col.lower():
            halka_aciklik_col = col
            break

    if halka_aciklik_col is None:
        st.warning("Halka açıklık verisi bulunamadı!")
        return {}

    # Create dictionary mapping Kod to Halka Açıklık
    try:
        halka_aciklik_dict = {}
        for _, row in df_ozet.iterrows():
            kod = row.get('Kod')
            halka_aciklik = row.get(halka_aciklik_col)

            if kod and pd.notna(halka_aciklik):
                # Try to convert to float, handling different formats
                try:
                    if isinstance(halka_aciklik, str):
                        # Remove % sign and convert
                        halka_aciklik = halka_aciklik.replace('%', '').replace(',', '.')
                    halka_aciklik_dict[str(kod).strip().upper()] = float(halka_aciklik)
                except (ValueError, TypeError):
                    continue

        return halka_aciklik_dict

    except Exception as e:
        st.warning(f"Halka açıklık verileri işlenirken hata: {e}")
        return {}


@st.cache_data
def load_lot_data():
    """Updated function to get circulation data from scraped DataFrame"""
    df_ozet = load_scraped_data()

    if df_ozet is None:
        return {}

    # Look for circulation amount column
    dolasim_col = None
    for col in df_ozet.columns:
        if 'fiili' in col.lower() and 'dolaşım' in col.lower() and 'tutar' in col.lower():
            dolasim_col = col
            break

    if dolasim_col is None:
        st.warning("Dolaşımdaki pay tutarı verisi bulunamadı!")
        return {}

    # Create dictionary mapping Kod to circulation amount
    try:
        dolasim_dict = {}
        for _, row in df_ozet.iterrows():
            kod = row.get('Kod')
            dolasim = row.get(dolasim_col)

            if kod and pd.notna(dolasim):
                # Try to convert to float, handling different formats
                try:
                    if isinstance(dolasim, str):
                        # Remove commas and convert
                        dolasim = dolasim.replace(',', '').replace('.', '')
                    dolasim_dict[str(kod).strip().upper()] = float(dolasim)
                except (ValueError, TypeError):
                    continue

        return dolasim_dict

    except Exception as e:
        st.warning(f"Dolaşım verileri işlenirken hata: {e}")
        return {}


# Load data
halka_aciklik_dict = load_halaciklik_data()
dolasim_lot_dict = load_lot_data()


# ---------------------- Teknik hesaplamalar ----------------------

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


# ---------------------- Multi-threaded Chart Preparation ----------------------

def prepare_single_chart_data(ticker):
    """
    Prepare chart data for a single ticker - designed for multi-threading
    """
    try:
        max_retries = 3
        data = None

        for attempt in range(max_retries):
            try:
                data = yf.download(ticker, period="1y", interval="1d", progress=False)
                if not data.empty and len(data) >= 50:
                    break
                time.sleep(0.1)
            except Exception:
                if attempt == max_retries - 1:
                    return ticker, None
                time.sleep(0.2)

        if data is None or data.empty or len(data) < 50:
            return ticker, None

        data.dropna(inplace=True)

        # Calculate technical indicators
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["MA200"] = data["Close"].rolling(200).mean()
        data["EMA89"] = data["Close"].ewm(span=89, adjust=False).mean()
        data["RSI"] = calculate_rsi(data["Close"])

        macd_line, signal_line, histogram = calculate_macd(data["Close"])
        data["MACD_Line"] = macd_line
        data["MACD_Signal"] = signal_line
        data["MACD_Hist"] = histogram

        return ticker, data

    except Exception:
        return ticker, None


def prepare_multiple_charts_data(tickers, max_workers=5):
    """
    Prepare chart data for multiple tickers using multi-threading
    """
    chart_data = {}
    completed_count = 0
    total_count = len(tickers)

    if total_count == 0:
        return chart_data

    # Create progress indicators for chart preparation
    chart_progress_bar = st.progress(0)
    chart_status_text = st.empty()

    # Thread-safe lock for updating progress
    progress_lock = threading.Lock()

    def update_chart_progress():
        nonlocal completed_count
        with progress_lock:
            completed_count += 1
            progress = completed_count / total_count
            chart_progress_bar.progress(progress)
            chart_status_text.text(f"Grafikler hazırlanıyor... {completed_count}/{total_count} grafik tamamlandı")

    # Use ThreadPoolExecutor for parallel chart data preparation
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(prepare_single_chart_data, ticker): ticker
            for ticker in tickers
        }

        # Collect results as they complete
        for future in as_completed(future_to_ticker):
            try:
                ticker, data = future.result()
                if data is not None:
                    chart_data[ticker] = data
                update_chart_progress()
            except Exception:
                update_chart_progress()

    # Clean up progress indicators
    chart_progress_bar.empty()
    chart_status_text.empty()

    return chart_data


def plot_stock_chart(data, ticker_name):
    """
    Plot stock chart with technical indicators
    """
    try:
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

    except Exception as e:
        st.error(f"Grafik çizilemedi: {str(e)}")
        plt.clf()


# ---------------------- Multi-threaded Stock Analysis ----------------------

def analyze_single_stock(ticker, ma_tolerance, volume_threshold, use_ma, use_volume,
                         use_rsi, rsi_threshold, ceiling_threshold):
    """
    Analyze a single stock - designed for multi-threading
    Returns dict with stock analysis or None if stock doesn't meet criteria
    """
    try:
        # Download stock data with retries
        max_retries = 3
        data = None

        for attempt in range(max_retries):
            try:
                data = yf.download(ticker, period="90d", interval="1d", progress=False)
                if not data.empty and len(data) >= 30:
                    break
                time.sleep(0.1)  # Small delay between retries
            except Exception:
                if attempt == max_retries - 1:
                    return None
                time.sleep(0.2)

        if data is None or data.empty or len(data) < 30:
            return None

        data.dropna(inplace=True)

        # Calculate technical indicators
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["MA200"] = data["Close"].rolling(200).mean()
        data["AvgVolume20"] = data["Volume"].rolling(20).mean()
        data["RSI"] = calculate_rsi(data["Close"])

        # Get latest values
        close = float(data["Close"].iloc[-1])
        prev_close = float(data["Close"].iloc[-2])
        change_pct = ((close - prev_close) / prev_close) * 100

        # Apply ceiling filter first (if enabled)
        if ceiling_threshold is not None and change_pct < ceiling_threshold:
            return None

        ma20 = float(data["MA20"].iloc[-1])
        ma50 = float(data["MA50"].iloc[-1])
        ma200 = float(data["MA200"].iloc[-1])
        rsi_latest = data["RSI"].iloc[-1]
        last_date = data.index[-1].strftime("%Y-%m-%d")
        volume = int(data["Volume"].iloc[-1])
        avg_volume = float(data["AvgVolume20"].iloc[-1])
        volume_ratio = volume / avg_volume if avg_volume > 0 else 0

        # Apply filters
        is_near_ma = close < min(ma20, ma50, ma200) * (1 + ma_tolerance)
        passes_ma = is_near_ma if use_ma else True
        passes_volume = volume_ratio >= volume_threshold if use_volume else True
        passes_rsi = rsi_latest <= rsi_threshold if use_rsi else True

        # Return result if all criteria are met
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

    except Exception:
        return None


def scan_stocks(tickers, ma_tolerance, volume_threshold, use_ma, use_volume,
                use_rsi=False, rsi_threshold=30, ceiling_threshold=None, max_workers=10):
    """
    Multi-threaded stock scanning function
    """
    results = []
    completed_count = 0
    total_count = len(tickers)

    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Thread-safe lock for updating progress
    progress_lock = threading.Lock()

    def update_progress():
        nonlocal completed_count
        with progress_lock:
            completed_count += 1
            progress = completed_count / total_count
            progress_bar.progress(progress)
            status_text.text(f"Tarama devam ediyor... {completed_count}/{total_count} hisse tamamlandı")

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(
                analyze_single_stock,
                ticker, ma_tolerance, volume_threshold,
                use_ma, use_volume, use_rsi, rsi_threshold, ceiling_threshold
            ): ticker for ticker in tickers
        }

        # Collect results as they complete
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                update_progress()
            except Exception:
                # Log error but continue processing
                update_progress()

    # Clean up progress indicators
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

# Performance settings
st.sidebar.subheader("🚀 Performans Ayarları")
max_workers_scan = st.sidebar.slider("Tarama - Eşzamanlı İşlem Sayısı", 5, 20, 10,
                                     help="Daha yüksek değerler daha hızlı tarama sağlar")
max_workers_chart = st.sidebar.slider("Grafik - Eşzamanlı İşlem Sayısı", 3, 10, 5,
                                      help="Grafikleri paralel hazırlar")

# Hisse Seçimi - Updated to use scraped data
all_tickers = get_all_bist_tickers()
selected_tickers = st.sidebar.multiselect("📌 Tarama İçin Hisse Seç (boş bırak tümü için)", options=all_tickers)

# ---------------------- Ana içerik ----------------------

if st.button("🔍 Taramayı Başlat"):
    with st.spinner("Hisseler taranıyor..."):
        tickers_to_scan = selected_tickers if selected_tickers else all_tickers

        # Add .IS suffix for yfinance
        tickers_with_suffix = [ticker + ".IS" if not ticker.endswith(".IS") else ticker for ticker in tickers_to_scan]

        ceiling_threshold = 9.5 if use_ceiling_filter else None

        # Display scan information
        st.info(f"📊 Toplam {len(tickers_with_suffix)} hisse taranacak. "
                f"Tarama: {max_workers_scan}, Grafik: {max_workers_chart} eşzamanlı işlem kullanılıyor.")

        start_time = time.time()

        # Run multi-threaded scan
        df = scan_stocks(
            tickers_with_suffix, ma_tolerance, volume_threshold,
            use_ma, use_volume, use_rsi, rsi_threshold, ceiling_threshold,
            max_workers=max_workers_scan
        )

        end_time = time.time()
        scan_duration = end_time - start_time

        if df.empty:
            st.warning("❌ Kriterlere uyan hisse bulunamadı.")
            st.info(f"⏱️ Tarama süresi: {scan_duration:.2f} saniye")
        else:
            st.success(f"✅ {len(df)} hisse bulundu. Tarama süresi: {scan_duration:.2f} saniye")

            # Sort results by RSI
            df_sorted = df.sort_values('RSI', ascending=True)

            # Prepare chart data for all found stocks in parallel
            chart_start_time = time.time()
            chart_tickers = [row['Hisse'] + ".IS" for _, row in df_sorted.iterrows()]

            st.info("📈 Grafikler hazırlanıyor...")
            chart_data_dict = prepare_multiple_charts_data(chart_tickers, max_workers=max_workers_chart)

            chart_end_time = time.time()
            chart_duration = chart_end_time - chart_start_time

            st.success(f"📊 Grafikler hazırlandı. Grafik hazırlama süresi: {chart_duration:.2f} saniye")
            st.info(f"🚀 Toplam süre: {scan_duration + chart_duration:.2f} saniye")

            # Display results with pre-loaded charts
            for _, row in df_sorted.iterrows():
                hisse = row['Hisse']
                ticker_full = hisse + ".IS"

                # Get additional info
                info = {}
                try:
                    info = yf.Ticker(ticker_full).info
                except:
                    pass

                market_cap_try = info.get("marketCap", None)
                usdtry = None
                try:
                    usdtry = yf.Ticker("USDTRY=X").info.get("regularMarketPrice", None)
                except:
                    pass

                market_cap_usd_str = "N/A"
                if market_cap_try and usdtry:
                    market_cap_usd = market_cap_try / usdtry
                    if market_cap_usd >= 1e9:
                        market_cap_usd_str = f"{market_cap_usd / 1e9:.2f} Milyar $"
                    elif market_cap_usd >= 1e6:
                        market_cap_usd_str = f"{market_cap_usd / 1e6:.2f} Milyon $"

                # Get lot data from scraped data
                lot = dolasim_lot_dict.get(hisse, "N/A")
                if lot != "N/A":
                    try:
                        lot = f"{int(float(lot)):,}".replace(",", ".")
                    except (ValueError, TypeError):
                        lot = str(lot)

                # Get halka açıklık data from scraped data
                halka_aciklik = halka_aciklik_dict.get(hisse, "N/A")
                if halka_aciklik != "N/A":
                    try:
                        halka_aciklik = f"%{float(halka_aciklik):.2f}"
                    except (ValueError, TypeError):
                        halka_aciklik = str(halka_aciklik)

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
                    <b>Halka Açıklık Oranı:</b> {halka_aciklik}<br><br>
                    📊 <b>Finansal Oranlar</b><br>
                    F/K: <b>{info.get("trailingPE", "N/A")}</b><br>
                    PD/DD: <b>{info.get("priceToBook", "N/A")}</b><br>
                    Piyasa Değeri: <b>{market_cap_usd_str}</b>
                </div>
                """, unsafe_allow_html=True)

                # Use pre-loaded chart data
                if ticker_full in chart_data_dict:
                    plot_stock_chart(chart_data_dict[ticker_full], hisse)
                else:
                    st.info(f"📊 {hisse} için grafik verisi hazırlanamadı.")