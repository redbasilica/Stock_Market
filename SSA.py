import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import warnings
from datetime import datetime
import io

warnings.filterwarnings('ignore')

# Import functions from d_scrap
from d_scrap import scrape_stock_data, scrape_kap_data, update_dataframe_types

st.set_page_config(page_title="BIST Hisse Analiz", layout="wide")

# Custom CSS for better UI
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: 500;
    }
    .help-button {
        background-color: #f0f2f6;
        border: 1px solid #e0e0e0;
    }
    .main-container {
        padding: 1rem;
    }
    div[data-testid="column"] {
        padding: 0 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("📈 BIST Hisse Analiz Platformu")

# ---------------------- Cache Management ----------------------

# Session state for caching downloaded data
if 'stock_data_cache' not in st.session_state:
    st.session_state.stock_data_cache = {}
if 'ticker_info_cache' not in st.session_state:
    st.session_state.ticker_info_cache = {}
if 'show_help' not in st.session_state:
    st.session_state.show_help = False


# ---------------------- Help Panel ----------------------

def toggle_help():
    st.session_state.show_help = not st.session_state.show_help


# Help button in the top right
col_main, col_help = st.columns([10, 1])
with col_help:
    if st.button("❓", key="help_btn", help="Kullanım Kılavuzu"):
        toggle_help()

if st.session_state.show_help:
    with st.container():
        st.info("""
        ### 📖 Kullanım Kılavuzu

        **1. Filtre Ayarları**
        - **MA Yakınlık Toleransı**: Hissenin hareketli ortalamalara ne kadar yakın olması gerektiğini belirler
        - **Hacim Artış Eşiği**: Normal hacmin kaç katı işlem görmesi gerektiğini belirler
        - **RSI Dip Seviyesi**: RSI göstergesinin maksimum değerini belirler (opsiyonel)
        - **Tavan Filtresi**: Bugün %9.5 ve üzeri artış gösteren hisseleri filtreler

        **2. Hisse Seçimi**
        - Belirli hisseleri taramak için listeden seçin
        - Boş bırakırsanız tüm hisseler taranır

        **3. Tarama**
        - "Taramayı Başlat" butonuna tıklayın
        - Sonuçlar otomatik olarak listelenecektir

        **4. Sonuçlar**
        - Her hisse için detaylı bilgiler görüntülenir
        - Grafik göster seçeneği ile teknik analiz grafikleri incelenebilir
        - Sonuçları Excel formatında indirebilirsiniz
        """)
        if st.button("✖️ Kapat", key="close_help"):
            st.session_state.show_help = False
            st.rerun()


# ---------------------- Veri yükleme ----------------------

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_stock_data():
    """Load and process stock data from web scraping"""
    try:
        # Scrape data from İş Yatırım
        df_ozet = scrape_stock_data()
        if df_ozet is None:
            st.error("İş Yatırım verisini yüklerken hata oluştu!")
            return None, {}, {}

        # Scrape data from KAP
        df_temp = scrape_kap_data()

        # Update df_ozet with KAP data
        df_ozet = update_dataframe_types(df_ozet, df_temp)

        # Clean column names and prepare data
        df_ozet.columns = df_ozet.columns.str.strip()

        # Get all tickers from 'Kod' column
        if 'Kod' in df_ozet.columns:
            all_tickers = [ticker.strip().upper() + ".IS" for ticker in df_ozet['Kod'].dropna().astype(str)]
        else:
            st.error("'Kod' sütunu bulunamadı!")
            return None, {}, {}

        # Prepare halka açıklık dictionary
        halka_aciklik_dict = {}
        halka_aciklik_col = None

        # Try different possible column names for halka açıklık
        possible_halka_cols = ['Halka Açıklık Oranı (%)', 'Halka AçıklıkOranı (%)', 'Halka Açıklık (%)',
                               'HalkaAçıklık(%)']
        for col in possible_halka_cols:
            if col in df_ozet.columns:
                halka_aciklik_col = col
                break

        if halka_aciklik_col:
            df_ozet["Kod"] = df_ozet["Kod"].str.strip().str.upper()
            halka_aciklik_dict = df_ozet.set_index("Kod")[halka_aciklik_col].to_dict()

        # Prepare dolaşımdaki lot dictionary
        dolasim_lot_dict = {}
        dolasim_col = None

        # Try different possible column names for dolaşımdaki lot
        possible_lot_cols = ['Fiili Dolaşımdaki Pay Tutarı(TL)', 'Dolasimdaki_Lot', 'Dolaşımdaki Lot', 'Fiili Dolaşım']
        for col in possible_lot_cols:
            if col in df_ozet.columns:
                dolasim_col = col
                break

        if dolasim_col:
            dolasim_lot_dict = df_ozet.set_index("Kod")[dolasim_col].to_dict()

        return all_tickers, halka_aciklik_dict, dolasim_lot_dict

    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {str(e)}")
        return None, {}, {}


# ---------------------- Optimized Technical Calculations ----------------------

@lru_cache(maxsize=128)
def calculate_rsi_vectorized(close_prices, period=14):
    """Vectorized RSI calculation for better performance"""
    close_array = np.array(close_prices)
    delta = np.diff(close_array)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period).mean().values

    rs = avg_gain / (avg_loss + 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    # Pad the beginning to match original length
    rsi_full = np.concatenate([np.full(period, np.nan), rsi[:len(close_array) - period]])
    return rsi_full[-1] if len(rsi_full) > 0 else np.nan


def calculate_all_indicators(data):
    """Calculate all technical indicators at once"""
    # Moving averages
    data["MA20"] = data["Close"].rolling(20, min_periods=1).mean()
    data["MA50"] = data["Close"].rolling(50, min_periods=1).mean()
    data["MA200"] = data["Close"].rolling(200, min_periods=1).mean()
    data["EMA89"] = data["Close"].ewm(span=89, adjust=False).mean()

    # Volume
    data["AvgVolume20"] = data["Volume"].rolling(20, min_periods=1).mean()

    # RSI
    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    data["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast = data["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD_Line"] = ema_fast - ema_slow
    data["MACD_Signal"] = data["MACD_Line"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD_Line"] - data["MACD_Signal"]

    return data


# ---------------------- Parallel Data Fetching ----------------------

def fetch_stock_data(ticker, period="90d", cache_dict=None):
    """Fetch stock data with optional caching"""
    cache_key = f"{ticker}_{period}"

    # Check cache if provided
    if cache_dict and cache_key in cache_dict:
        return ticker, cache_dict[cache_key]

    try:
        data = yf.download(ticker, period=period, interval="1d", progress=False, threads=False)
        if not data.empty and len(data) >= 30:
            data = calculate_all_indicators(data)
            # Store in cache if provided
            if cache_dict is not None:
                cache_dict[cache_key] = data
            return ticker, data
    except Exception:
        pass

    return ticker, None


def fetch_ticker_info(ticker, cache_dict=None):
    """Fetch ticker info with optional caching"""
    if cache_dict and ticker in cache_dict:
        return ticker, cache_dict[ticker]

    try:
        info = yf.Ticker(ticker).info
        if cache_dict is not None:
            cache_dict[ticker] = info
        return ticker, info
    except:
        return ticker, {}


# ---------------------- Optimized Scanning Function ----------------------

def scan_stocks_parallel(tickers, ma_tolerance, volume_threshold, use_ma, use_volume,
                         use_rsi=False, rsi_threshold=30, ceiling_threshold=None, max_workers=10):
    """Parallel stock scanning for better performance"""
    results = []

    # Get cache from session state - create a copy to avoid thread issues
    stock_cache = dict(st.session_state.stock_data_cache)

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks with cache dictionary
        futures = {executor.submit(fetch_stock_data, ticker, "90d", stock_cache): ticker
                   for ticker in tickers}

        completed = 0
        total = len(tickers)

        for future in as_completed(futures):
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"Taranıyor: {completed}/{total} hisse")

            ticker, data = future.result()

            if data is None or data.empty:
                continue

            try:
                # Get latest values
                close = float(data["Close"].iloc[-1])
                prev_close = float(data["Close"].iloc[-2])
                change_pct = ((close - prev_close) / prev_close) * 100

                # Apply ceiling filter if enabled
                if ceiling_threshold is not None and change_pct < ceiling_threshold:
                    continue

                ma20 = float(data["MA20"].iloc[-1])
                ma50 = float(data["MA50"].iloc[-1])
                ma200 = float(data["MA200"].iloc[-1]) if len(data) >= 200 else ma50
                rsi_latest = float(data["RSI"].iloc[-1])
                last_date = data.index[-1].strftime("%Y-%m-%d")
                volume = int(data["Volume"].iloc[-1])
                avg_volume = float(data["AvgVolume20"].iloc[-1])
                volume_ratio = volume / avg_volume if avg_volume > 0 else 0

                # Apply filters
                is_near_ma = close < min(ma20, ma50, ma200) * (1 + ma_tolerance)
                passes_ma = is_near_ma if use_ma else True
                passes_volume = volume_ratio >= volume_threshold if use_volume else True
                passes_rsi = rsi_latest <= rsi_threshold if use_rsi else True

                if passes_ma and passes_volume and passes_rsi:
                    results.append({
                        "Hisse": ticker.replace(".IS", ""),
                        "Tarih": last_date,
                        "Kapanış": round(close, 2),
                        "Değişim": round(change_pct, 2),
                        "MA20": round(ma20, 2),
                        "MA50": round(ma50, 2),
                        "Hacim Katsayısı": round(volume_ratio, 2),
                        "RSI": round(rsi_latest, 2)
                    })
            except Exception:
                continue

    # Update session state cache with new data
    st.session_state.stock_data_cache.update(stock_cache)

    progress_bar.empty()
    status_text.empty()

    return pd.DataFrame(results)


# ---------------------- Optimized Plotting ----------------------

@st.cache_data
def prepare_plot_data(ticker):
    """Prepare data for plotting with caching"""
    data = yf.download(ticker, period="1y", interval="1d", progress=False, threads=False)
    if data.empty or len(data) < 50:
        return None

    data = calculate_all_indicators(data)
    return data


def plot_stock_chart(data, ticker_name):
    """Optimized plotting function"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=True,
                                        gridspec_kw={'height_ratios': [2, 1, 1]})

    # Main price chart
    ax1.plot(data.index, data["Close"], label="Kapanış", color="blue", linewidth=1.5)
    ax1.plot(data.index, data["MA20"], label="MA20", color="orange", linewidth=1)
    ax1.plot(data.index, data["MA50"], label="MA50", color="green", linewidth=1)

    if len(data) >= 200:
        ax1.plot(data.index, data["MA200"], label="MA200", color="red", linewidth=1)

    ax1.plot(data.index, data["EMA89"], label="EMA89", color="magenta", linestyle="--", linewidth=1)
    ax1.set_title(f"{ticker_name} - Son 1 Yıl Teknik Görünüm")
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # RSI
    ax2.plot(data.index, data["RSI"], label="RSI", color="purple", linewidth=1)
    ax2.axhline(70, color='red', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_ylabel("RSI")
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # MACD
    ax3.plot(data.index, data["MACD_Line"], label="MACD", color="blue", linewidth=1)
    ax3.plot(data.index, data["MACD_Signal"], label="Signal", color="orange", linewidth=1)
    ax3.bar(data.index, data["MACD_Hist"], label="Histogram", color="gray", alpha=0.4)
    ax3.set_ylabel("MACD")
    ax3.legend(loc='best', fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Watermark
    fig.text(0.5, 0.5, 'Bay-P',
             fontsize=50, color='gray', alpha=0.15,
             ha='center', va='center',
             weight='bold', style='italic', rotation=20)

    plt.tight_layout()
    st.pyplot(fig)
    plt.clf()
    plt.close()  # Explicitly close the figure to free memory


# ---------------------- Load Initial Data ----------------------

with st.spinner("Veriler yükleniyor..."):
    all_tickers, halka_aciklik_dict, dolasim_lot_dict = load_stock_data()

if all_tickers is None:
    st.stop()

# ---------------------- Main Control Panel ----------------------

st.markdown("---")

# Filter Settings Section
st.subheader("⚙️ Filtre Ayarları")

# Group 1: MA and Volume Filters
col1, col2, col3, col4 = st.columns(4)

with col1:
    use_ma = st.checkbox("📊 MA Dip Filtresi", value=True, help="Hareketli ortalama dip filtresi")
    ma_tolerance = st.slider("MA Yakınlık (%)", 1, 10, 5, disabled=not use_ma) / 100

with col2:
    use_volume = st.checkbox("📈 Hacim Filtresi", value=True, help="Hacim artış filtresi")
    volume_threshold = st.slider("Hacim Artışı (kat)", 0.0, 5.0, 1.5, disabled=not use_volume)

with col3:
    use_rsi = st.checkbox("📉 RSI Filtresi", value=False, help="RSI dip filtresi")
    rsi_threshold = st.slider("RSI Eşiği", 10, 50, 30, disabled=not use_rsi)

with col4:
    use_ceiling_filter = st.checkbox("🚀 Tavan Filtresi", value=False, help="Bugün %9.5+ artanları filtrele")
    st.markdown("")  # Spacer
    st.markdown("")  # Spacer

st.markdown("---")

# Stock Selection Section
st.subheader("📌 Hisse Seçimi")

col1, col2 = st.columns([3, 1])

with col1:
    ticker_options = sorted([ticker.replace(".IS", "") for ticker in all_tickers])
    selected_tickers = st.multiselect(
        "Taranacak hisseleri seçin (boş = tüm hisseler)",
        options=ticker_options,
        placeholder="Hisse seçin veya tümünü taramak için boş bırakın"
    )

with col2:
    st.metric("📊 Toplam Hisse", len(all_tickers))
    st.metric("✅ Seçili", len(selected_tickers) if selected_tickers else "Tümü")

# Convert selected tickers back to .IS format
selected_tickers_full = [ticker + ".IS" for ticker in selected_tickers] if selected_tickers else []

st.markdown("---")

# Action Buttons
col1, col2, col3 = st.columns([2, 1, 1])

with col1:
    scan_button = st.button("🔍 Taramayı Başlat", type="primary", use_container_width=True)

with col2:
    if st.button("🗑️ Önbelleği Temizle", use_container_width=True):
        st.session_state.stock_data_cache = {}
        st.session_state.ticker_info_cache = {}
        st.cache_data.clear()
        st.success("✅ Önbellek temizlendi!")

with col3:
    # Empty column for spacing
    pass

# ---------------------- Main Results Section ----------------------

if scan_button:
    with st.spinner("Hisseler taranıyor..."):
        tickers_to_scan = selected_tickers_full if selected_tickers_full else all_tickers
        ceiling_threshold = 9.5 if use_ceiling_filter else None
        max_workers = 10  # Fixed value instead of user-configurable

        # Run parallel scanning
        df = scan_stocks_parallel(
            tickers_to_scan,
            ma_tolerance,
            volume_threshold,
            use_ma,
            use_volume,
            use_rsi,
            rsi_threshold,
            ceiling_threshold,
            max_workers
        )

        if df.empty:
            st.warning("⚠️ Kriterlere uyan hisse bulunamadı.")
        else:
            st.success(f"✅ {len(df)} hisse bulundu.")

            # Fetch all ticker info in parallel for found stocks
            info_dict = {}
            info_cache = dict(st.session_state.ticker_info_cache)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(fetch_ticker_info, row['Hisse'] + ".IS", info_cache): row['Hisse']
                           for _, row in df.iterrows()}

                for future in as_completed(futures):
                    ticker, info = future.result()
                    info_dict[ticker] = info

            # Update session state cache
            st.session_state.ticker_info_cache.update(info_cache)

            # Also fetch USD/TRY rate once
            usdtry = None
            try:
                usdtry = yf.Ticker("USDTRY=X").info.get("regularMarketPrice", None)
            except:
                pass

            # Display results
            st.markdown("---")
            st.subheader("📊 Tarama Sonuçları")

            for idx, row in df.iterrows():
                hisse = row['Hisse']
                ticker_full = hisse + ".IS"
                info = info_dict.get(ticker_full, {})

                # Calculate market cap
                market_cap_try = info.get("marketCap", None)
                market_cap_usd_str = "N/A"
                if market_cap_try and usdtry:
                    market_cap_usd = market_cap_try / usdtry
                    if market_cap_usd >= 1e9:
                        market_cap_usd_str = f"{market_cap_usd / 1e9:.2f} Milyar $"
                    elif market_cap_usd >= 1e6:
                        market_cap_usd_str = f"{market_cap_usd / 1e6:.2f} Milyon $"

                # Get lot information
                lot = dolasim_lot_dict.get(hisse, "N/A")
                if lot != "N/A" and pd.notna(lot):
                    try:
                        lot = f"{int(lot):,}".replace(",", ".")
                    except:
                        lot = str(lot)

                # Get halka açıklık information
                halka_aciklik = halka_aciklik_dict.get(hisse, "N/A")
                if halka_aciklik != "N/A" and pd.notna(halka_aciklik):
                    try:
                        if isinstance(halka_aciklik, (int, float)) and halka_aciklik <= 1:
                            halka_aciklik = f"%{halka_aciklik * 100:.2f}"
                        else:
                            halka_aciklik = f"%{float(halka_aciklik):.2f}"
                    except:
                        halka_aciklik = str(halka_aciklik)

                # Display info card
                color = "green" if row['Değişim'] >= 0 else "red"
                sign = "▲" if row['Değişim'] >= 0 else "▼"

                with st.expander(
                        f"**{hisse}** - {row['Tarih']} | Kapanış: {row['Kapanış']} {sign} {abs(row['Değişim'])}%",
                        expanded=(idx < 3)):  # Expand first 3 results

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.markdown(f"""
                        **Fiyat Bilgileri**
                        - Kapanış: **{row['Kapanış']}**
                        - Değişim: <span style='color:{color}'>{sign} {abs(row['Değişim'])}%</span>
                        - MA20: {row['MA20']}
                        - MA50: {row['MA50']}
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                        **Teknik Göstergeler**
                        - RSI: **{row['RSI']}**
                        - Hacim/Ort: **{row['Hacim Katsayısı']}**
                        - Dolaşımdaki Lot: {lot}
                        - Halka Açıklık: {halka_aciklik}
                        """)

                    with col3:
                        pe_ratio = info.get("trailingPE", "N/A")
                        pb_ratio = info.get("priceToBook", "N/A")

                        # Format ratios
                        if isinstance(pe_ratio, (int, float)):
                            pe_ratio = f"{pe_ratio:.2f}"
                        if isinstance(pb_ratio, (int, float)):
                            pb_ratio = f"{pb_ratio:.2f}"

                        st.markdown(f"""
                        **Finansal Oranlar**
                        - F/K: **{pe_ratio}**
                        - PD/DD: **{pb_ratio}**
                        - Piyasa Değeri: **{market_cap_usd_str}**
                        """)

                    # Show chart
                    if st.checkbox(f"📊 Grafik Göster", key=f"chart_{hisse}"):
                        data_plot = prepare_plot_data(ticker_full)
                        if data_plot is not None:
                            plot_stock_chart(data_plot, hisse)
                        else:
                            st.info(f"{hisse} için yeterli veri bulunamadı.")

            # Export functionality
            st.markdown("---")
            st.subheader("💾 Dışa Aktarma")

            col1, col2, col3 = st.columns([1, 1, 2])

            # Get current date for filename
            current_date = datetime.now().strftime("%Y%m%d")

            with col1:
                # Export as CSV
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📄 CSV İndir",
                    data=csv,
                    file_name=f'tarama_sonuclari_{current_date}.csv',
                    mime='text/csv',
                    use_container_width=True
                )

            with col2:
                # Export as Excel
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Tarama Sonuçları', index=False)

                    # Auto-adjust columns width
                    worksheet = writer.sheets['Tarama Sonuçları']
                    for column in worksheet.columns:
                        max_length = 0
                        column_letter = column[0].column_letter
                        for cell in column:
                            try:
                                if len(str(cell.value)) > max_length:
                                    max_length = len(str(cell.value))
                            except:
                                pass
                        adjusted_width = min(max_length + 2, 50)
                        worksheet.column_dimensions[column_letter].width = adjusted_width

                excel_data = output.getvalue()

                st.download_button(
                    label="📊 Excel İndir",
                    data=excel_data,
                    file_name=f'tarama_sonuclari_{current_date}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    use_container_width=True
                )

            with col3:
                st.info(f"📅 Tarama Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <small>BIST Hisse Analiz Platformu v1.2 | Bay-P</small>
    </div>
    """,
    unsafe_allow_html=True
)