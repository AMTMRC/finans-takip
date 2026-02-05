import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from prophet import Prophet

# --- AYARLAR ---
st.set_page_config(page_title="Finans Kahini", layout="wide")

# Takip edilecek semboller
semboller = {
    'USD/TRY (Dolar)': 'USDTRY=X',
    'EUR/TRY (Euro)': 'EURTRY=X',
    'GBP/TRY (Sterlin)': 'GBPTRY=X',
    'Gram Altın (Tahmini)': 'GC=F', 
    'Gümüş (Ons)': 'SI=F',
    'Bitcoin ($)': 'BTC-USD',
    'Ethereum ($)': 'ETH-USD',
    'BIST 100 (Endeks)': 'XU100.IS'
}

# --- YAN MENÜ ---
st.sidebar.title("🔮 Finans Kahini")
secilen_isim = st.sidebar.selectbox("Analiz Edilecek Varlık", list(semboller.keys()))
secilen_sembol = semboller[secilen_isim]

# Prophet için daha fazla veri lazım, o yüzden varsayılan olarak uzun dönem çekiyoruz
st.sidebar.info("Yapay zeka tahmini için son 2 yılın verisi baz alınır.")

# --- VERİ ÇEKME FONKSİYONU ---
@st.cache_data(ttl=300)
def veri_getir(sembol):
    # Tahmin için en az 2 yıllık veri çekiyoruz
    df = yf.download(tickers=sembol, period="2y", interval="1d", progress=False)
    
    # Sütun düzeltme
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.get_level_values(0)
        except:
            pass
            
    # Gram altın hesabı
    if sembol == 'GC=F': 
        usd_data = yf.download('USDTRY=X', period='1d', interval='1m', progress=False)
        if isinstance(usd_data.columns, pd.MultiIndex):
            usd_data.columns = usd_data.columns.get_level_values(0)
        usd_try = float(usd_data['Close'].iloc[-1])
        df['Close'] = (df['Close'] * usd_try) / 31.1035
        df['Open'] = (df['Open'] * usd_try) / 31.1035
        
    return df

# --- ANA EKRAN ---
st.title(f"📈 {secilen_isim} & Yapay Zeka Tahmini")

try:
    with st.spinner('Piyasa verileri ve Yapay Zeka modelleri çalıştırılıyor...'):
        df = veri_getir(secilen_sembol)
    
    if not df.empty and 'Close' in df.columns:
        son_fiyat = float(df['Close'].iloc[-1])
        onceki_fiyat = float(df['Close'].iloc[-2])
        degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
        
        # Metrikler
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Anlık Fiyat", value=f"{son_fiyat:.2f}", delta=f"%{degisim:.2f}")
        col2.info("Tahmin Modeli: Facebook Prophet")
        
        # --- PROPHET TAHMİN MOTORU ---
        # Prophet 'ds' (tarih) ve 'y' (değer) sütunları ister
        df_prophet = df.reset_index()[['Date', 'Close']]
        df_prophet.columns = ['ds', 'y']
        
        # Modeli eğit
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        
        # Gelecek 30 gün için boş tablo oluştur
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        
        # --- GRAFİK ÇİZİMİ ---
        fig = go.Figure()

        # 1. Gerçek Veriler (Geçmiş)
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Gerçekleşen Fiyat', line=dict(color='cyan', width=2)))

        # 2. Tahmin Verileri (Gelecek)
        # Sadece geleceği çizdirmek için son 30 günü filtreleyelim
        future_forecast = forecast.tail(30)
        fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], name='Yapay Zeka Tahmini', line=dict(color='orange', width=2, dash='dot')))

        # 3. Alt ve Üst Sınırlar (Güven Aralığı)
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'], y=future_forecast['yhat_upper'],
            mode='lines', line=dict(width=0), showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'], y=future_forecast['yhat_lower'],
            mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', name='Tahmin Aralığı'
        ))

        fig.update_layout(
            title=f'{secilen_isim} - 30 Günlük Gelecek Tahmini',
            yaxis_title='Fiyat',
            template="plotly_dark",
            height=600,
            legend=dict(orientation="h", y=1.02, x=0.8)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Yorum
        tahmin_son = future_forecast['yhat'].iloc[-1]
        fark_tahmin = tahmin_son - son_fiyat
        
        st.subheader("📢 Yapay Zeka Yorumu")
        if fark_tahmin > 0:
            st.success(f"Yapay zeka, önümüzdeki 30 gün içinde fiyatın **YÜKSELECEĞİNİ** öngörüyor. (Hedef: {tahmin_son:.2f})")
        else:
            st.error(f"Yapay zeka, önümüzdeki 30 gün içinde fiyatın **DÜŞEBİLECEĞİNİ** veya yatay gideceğini öngörüyor. (Hedef: {tahmin_son:.2f})")

        with st.expander("Tahmin Verilerini İncele"):
            st.dataframe(future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10))

    else:
        st.error("Veri alınamadı. Lütfen daha sonra tekrar deneyin.")

except Exception as e:
    st.error(f"Bir hata oluştu: {e}")