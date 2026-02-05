import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots # Yeni eklendi (Alt alta grafik için)
from prophet import Prophet
from datetime import datetime, timedelta

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Finans Pro v3.0", layout="wide", initial_sidebar_state="expanded")

# --- CSS ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #0e1117; border-radius: 5px; color: white; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #262730; color: #FFA500; }
    .stMetric { background-color: #1f2937; border: 1px solid #374151; padding: 15px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- VARLIK LİSTESİ ---
varliklar = {
    'USD - Amerikan Doları': {'ticker': 'USDTRY=X', 'source': 'direct'},
    'EUR - Avrupa Para Birimi': {'ticker': 'EURTRY=X', 'source': 'direct'},
    'GBP - İngiliz Sterlini': {'ticker': 'GBPTRY=X', 'source': 'direct'},
    'XAU - Altın (Gram)': {'ticker': 'GC=F', 'source': 'gold_calc'},
    'XAG - Gümüş (Gram)': {'ticker': 'SI=F', 'source': 'silver_calc'},
    'CAD - Kanada Doları': {'ticker': 'USDCAD=X', 'source': 'calc'},
    'CHF - İsviçre Frangı': {'ticker': 'CHF=X', 'source': 'calc_inverse'},
    'AUD - Avustralya Doları': {'ticker': 'AUDUSD=X', 'source': 'calc_multiply'},
    'DKK - Danimarka Kronu': {'ticker': 'USDDKK=X', 'source': 'calc'},
    'JPY - Japon Yeni': {'ticker': 'USDJPY=X', 'source': 'calc'},
    'KWD - Kuveyt Dinarı': {'ticker': 'USDKWD=X', 'source': 'calc'}, 
    'NOK - Norveç Kronu': {'ticker': 'USDNOK=X', 'source': 'calc'},
    'SAR - Suudi Arabistan Riyali': {'ticker': 'USDSAR=X', 'source': 'calc'},
    'SEK - İsveç Kronu': {'ticker': 'USDSEK=X', 'source': 'calc'},
    'AED - Bae Dirhemi': {'ticker': 'USDAED=X', 'source': 'calc'},
    'AZN - Azerbaycan Manatı': {'ticker': 'USDAZN=X', 'source': 'calc'}, 
    'RON - Rumen Leyi': {'ticker': 'USDRON=X', 'source': 'calc'}
}

# --- YARDIMCI FONKSİYONLAR ---
# RSI Hesaplama Fonksiyonu (Teknik Analiz)
def rsi_hesapla(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=300)
def veri_getir(info):
    try:
        usd_try_df = yf.download('USDTRY=X', period="2y", interval="1d", progress=False)
        if isinstance(usd_try_df.columns, pd.MultiIndex): usd_try_df.columns = usd_try_df.columns.get_level_values(0)
        
        if usd_try_df.empty: return pd.DataFrame()

        if info['source'] == 'direct':
            df = yf.download(info['ticker'], period="2y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df

        elif info['source'] in ['gold_calc', 'silver_calc']:
            df = yf.download(info['ticker'], period="2y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.reindex(usd_try_df.index).dropna()
            eslesmis_usd = usd_try_df.reindex(df.index)
            for col in ['Close', 'Open', 'High', 'Low']:
                df[col] = (df[col] * eslesmis_usd['Close']) / 31.1035
            return df

        else:
            target_df = yf.download(info['ticker'], period="2y", interval="1d", progress=False)
            if isinstance(target_df.columns, pd.MultiIndex): target_df.columns = target_df.columns.get_level_values(0)
            common_index = usd_try_df.index.intersection(target_df.index)
            usd_try_filtered = usd_try_df.loc[common_index]
            target_filtered = target_df.loc[common_index]
            df = pd.DataFrame(index=common_index)
            
            if info['source'] == 'calc': 
                df['Close'] = usd_try_filtered['Close'] / target_filtered['Close']
            elif info['source'] == 'calc_inverse':
                df['Close'] = usd_try_filtered['Close'] * target_filtered['Close']
            elif info['source'] == 'calc_multiply':
                df['Close'] = usd_try_filtered['Close'] * target_filtered['Close']
            return df
    except:
        return pd.DataFrame()

# --- YAN MENÜ ---
st.sidebar.title("Finans Pro 🚀")
secilen_isim = st.sidebar.selectbox("Varlık Seçimi", list(varliklar.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Grafik Ayarları")
# Kullanıcı bu kutucukları işaretlerse grafik değişecek
goster_sma50 = st.sidebar.checkbox("50 Günlük Ortalama (Trend)", value=False)
goster_sma200 = st.sidebar.checkbox("200 Günlük Ortalama (Uzun Vade)", value=False)
goster_rsi = st.sidebar.checkbox("RSI Göstergesi (Ucuz/Pahalı)", value=True)

# --- VERİ ÇEKME ---
ana_df = veri_getir(varliklar[secilen_isim])

# --- SAYFA YAPISI ---
tab1, tab2, tab3 = st.tabs(["🔮 Analiz Merkezi", "🏎️ Yarış Pisti", "⏳ Zaman Makinesi"])

# ---------------------------------------------------------
# TAB 1: PROFESYONEL ANALİZ
# ---------------------------------------------------------
with tab1:
    st.header(f"📈 {secilen_isim} Teknik Analizi")
    
    if not ana_df.empty and len(ana_df) > 1:
        son_fiyat = float(ana_df['Close'].iloc[-1])
        onceki_fiyat = float(ana_df['Close'].iloc[-2])
        degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
        
        # Metrikler
        c1, c2, c3 = st.columns(3)
        c1.metric("Anlık Fiyat", f"{son_fiyat:.2f} ₺", f"%{degisim:.2f}")
        
        # RSI Hesapla ve Yorumla
        rsi_deger = rsi_hesapla(ana_df['Close']).iloc[-1]
        rsi_durum = "Nötr"
        rsi_renk = "off"
        if rsi_deger > 70: 
            rsi_durum = "⚠️ Aşırı Alım (Pahalı)" 
            rsi_renk = "normal"
        elif rsi_deger < 30: 
            rsi_durum = "✅ Aşırı Satım (Ucuz)"
            rsi_renk = "inverse"
        
        c2.metric("RSI İndikatörü", f"{rsi_deger:.1f}", rsi_durum, delta_color=rsi_renk)
        
        # Prophet Tahmini
        try:
            df_prophet = ana_df.reset_index()[['Date', 'Close']]
            df_prophet.columns = ['ds', 'y']
            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=14)
            forecast = model.predict(future)
            
            # --- GRAFİK OLUŞTURMA (SUBPLOTS) ---
            # Eğer RSI seçiliyse alt alta 2 grafik, değilse tek grafik
            rows = 2 if goster_rsi else 1
            row_heights = [0.7, 0.3] if goster_rsi else [1.0]
            vertical_gap = 0.1 if goster_rsi else 0
            
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, 
                                vertical_spacing=vertical_gap, row_heights=row_heights,
                                subplot_titles=(f"{secilen_isim} Fiyat Grafiği", "RSI Güç Endeksi") if goster_rsi else (f"{secilen_isim} Fiyat Grafiği",))

            # 1. Ana Fiyat Çizgisi
            gosterim_df = ana_df.tail(250) # Son 1 iş yılı
            fig.add_trace(go.Scatter(x=gosterim_df.index, y=gosterim_df['Close'], mode='lines', name='Fiyat', line=dict(color='#00BFFF', width=2)), row=1, col=1)

            # 2. Hareketli Ortalamalar (İsteğe Bağlı)
            if goster_sma50:
                sma50 = gosterim_df['Close'].rolling(window=50).mean()
                fig.add_trace(go.Scatter(x=gosterim_df.index, y=sma50, mode='lines', name='50 Günlük Ort.', line=dict(color='#ffff00', width=1)), row=1, col=1)
            
            if goster_sma200:
                sma200 = gosterim_df['Close'].rolling(window=200).mean()
                fig.add_trace(go.Scatter(x=gosterim_df.index, y=sma200, mode='lines', name='200 Günlük Ort.', line=dict(color='#ff00ff', width=1)), row=1, col=1)

            # 3. Tahmin Çizgisi
            future_forecast = forecast.tail(14)
            fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='YZ Tahmini', line=dict(color='#FFA500', width=2, dash='dot')), row=1, col=1)

            # 4. RSI Grafiği (Alt Kısım)
            if goster_rsi:
                rsi_series = rsi_hesapla(gosterim_df['Close'])
                fig.add_trace(go.Scatter(x=gosterim_df.index, y=rsi_series, mode='lines', name='RSI', line=dict(color='#9370DB')), row=2, col=1)
                # Sınır çizgileri (30 ve 70)
                fig.add_hrect(y0=70, y1=100, row=2, col=1, fillcolor="red", opacity=0.1, line_width=0)
                fig.add_hrect(y0=0, y1=30, row=2, col=1, fillcolor="green", opacity=0.1, line_width=0)
                fig.add_hline(y=70, line_dash="dot", row=2, col=1, line_color="red", line_width=1)
                fig.add_hline(y=30, line_dash="dot", row=2, col=1, line_color="green", line_width=1)

            fig.update_layout(template="plotly_dark", height=600, hovermode="x unified", legend=dict(orientation="h", y=1.02, x=0))
            st.plotly_chart(fig, use_container_width=True)
            
            # Yorum
            c3.info(f"Yapay Zeka Hedefi (14 Gün): {future_forecast['yhat'].iloc[-1]:.2f} TL")
            
        except Exception as e:
            st.warning("Teknik analiz grafiği oluşturulurken veri yetersiz kaldı.")
            st.line_chart(ana_df['Close'])
    else:
        st.error("Veri alınamadı.")

# ---------------------------------------------------------
# TAB 2: KARŞILAŞTIRMA MODU (Öncekiyle aynı)
# ---------------------------------------------------------
with tab2:
    st.header("🆚 Varlık Yarışı")
    karsilastirma_listesi = st.multiselect("Karşılaştırılacakları Seçin:", list(varliklar.keys()), default=['USD - Amerikan Doları', 'EUR - Avrupa Para Birimi', 'XAU - Altın (Gram)'])
    
    if karsilastirma_listesi:
        fig_comp = go.Figure()
        with st.spinner('Hesaplanıyor...'):
            for varlik in karsilastirma_listesi:
                df_temp = veri_getir(varliklar[varlik])
                if not df_temp.empty:
                    df_temp = df_temp.tail(365)
                    ilk_fiyat = float(df_temp['Close'].iloc[0])
                    df_temp['Normalize'] = (df_temp['Close'] / ilk_fiyat) * 100
                    fig_comp.add_trace(go.Scatter(x=df_temp.index, y=df_temp['Normalize'], mode='lines', name=varlik.split(' - ')[0]))
        
        fig_comp.update_layout(title="Son 1 Yıl Performansı (Başlangıç=100)", template="plotly_dark", height=500)
        st.plotly_chart(fig_comp, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: ZAMAN MAKİNESİ (Öncekiyle aynı)
# ---------------------------------------------------------
with tab3:
    st.header("⏳ Yatırım Simülatörü")
    col1, col2, col3 = st.columns(3)
    tutar = col1.number_input("Yatırılan Tutar (TL)", value=10000)
    tarih = col2.date_input("Hangi Tarihte?", value=datetime.now() - timedelta(days=365))
    varlik_secim = col3.selectbox("Hangi Varlık?", list(varliklar.keys()))
    
    if st.button("Hesapla"):
        df_sim = veri_getir(varliklar[varlik_secim])
        secilen_tarih_str = tarih.strftime('%Y-%m-%d')
        if not df_sim.empty:
            try:
                gecmis_veri = df_sim.iloc[df_sim.index.get_indexer([secilen_tarih_str], method='nearest')]
                gecmis_fiyat = float(gecmis_veri['Close'].values[0])
                guncel_fiyat = float(df_sim['Close'].iloc[-1])
                adet = tutar / gecmis_fiyat
                guncel_deger = adet * guncel_fiyat
                kar_zarar = guncel_deger - tutar
                
                c1, c2, c3 = st.columns(3)
                c1.metric("O Gün Alınan", f"{adet:.2f} Adet")
                c2.metric("Bugünkü Değer", f"{guncel_deger:,.2f} TL")
                c3.metric("Kar/Zarar", f"{kar_zarar:,.2f} TL", delta_color="normal" if kar_zarar > 0 else "inverse")
            except:
                st.error("Tarih verisi bulunamadı.")