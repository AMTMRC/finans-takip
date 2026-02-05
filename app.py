import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Finans Takip", layout="wide")

# --- LİSTE (Ekran Görüntüleri Baz Alındı) ---
# Yahoo Finance kodları ile eşleştirildi
semboller = {
    'USD - Amerikan Doları': 'USDTRY=X',
    'EUR - Avrupa Para Birimi': 'EURTRY=X',
    'GBP - İngiliz Sterlini': 'GBPTRY=X',
    'XAU - Altın (Gram)': 'GC=F',       # Özel hesaplama yapılacak
    'XAG - Gümüş (Gram)': 'SI=F',       # Özel hesaplama yapılacak
    'CAD - Kanada Doları': 'CADTRY=X',
    'CHF - İsviçre Frangı': 'CHFTRY=X',
    'AUD - Avustralya Doları': 'AUDTRY=X',
    'DKK - Danimarka Kronu': 'DKKTRY=X',
    'JPY - Japon Yeni': 'JPYTRY=X',
    'KWD - Kuveyt Dinarı': 'KWDTRY=X',
    'NOK - Norveç Kronu': 'NOKTRY=X',
    'SAR - Suudi Arabistan Riyali': 'SARTRY=X',
    'SEK - İsveç Kronu': 'SEKTRY=X',
    'AED - Bae Dirhemi': 'AEDTRY=X',
    'AZN - Azerbaycan Manatı': 'AZNTRY=X',
    'RON - Rumen Leyi': 'RONTRY=X'
}

# --- YAN MENÜ ---
st.sidebar.title("💰 Kur Seçimi")
secilen_isim = st.sidebar.selectbox("Para Birimi Seçiniz", list(semboller.keys()))
secilen_sembol = semboller[secilen_isim]

# --- VERİ ÇEKME FONKSİYONU ---
@st.cache_data(ttl=300)
def veri_getir(sembol):
    try:
        # Son 2 yılın verisini çekiyoruz ki grafik dolu dolu olsun
        df = yf.download(sembol, period="2y", interval="1d", progress=False)
        
        # Sütun isimlerini düzelt (MultiIndex sorunu için)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Eğer Altın veya Gümüş ise (ONS -> GRAM Çevirimi)
        if sembol in ['GC=F', 'SI=F']:
            usd_data = yf.download('USDTRY=X', period='1d', progress=False)
            if isinstance(usd_data.columns, pd.MultiIndex):
                usd_data.columns = usd_data.columns.get_level_values(0)
            usd_try = float(usd_data['Close'].iloc[-1])
            
            # Formül: (Ons Fiyatı * Dolar Kuru) / 31.1035
            df['Close'] = (df['Close'] * usd_try) / 31.1035
            df['Open'] = (df['Open'] * usd_try) / 31.1035
            df['High'] = (df['High'] * usd_try) / 31.1035
            df['Low'] = (df['Low'] * usd_try) / 31.1035

        return df
    except Exception as e:
        return pd.DataFrame()

# --- ANA EKRAN ---
st.title(f"📈 {secilen_isim}")

with st.spinner('Veriler güncelleniyor...'):
    df = veri_getir(secilen_sembol)

if not df.empty and 'Close' in df.columns:
    # Son fiyat ve değişim
    son_fiyat = float(df['Close'].iloc[-1])
    onceki_fiyat = float(df['Close'].iloc[-2])
    degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
    
    # Büyük Puntolu Fiyat Gösterimi
    st.metric(label="Anlık Değer (TL)", value=f"{son_fiyat:.2f} ₺", delta=f"%{degisim:.2f}")
    
    # --- PROPHET TAHMİNİ ---
    df_prophet = df.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    
    # Modeli kur ve eğit
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    
    # Gelecek 14 gün (2 Hafta) tahmini
    future = model.make_future_dataframe(periods=14)
    forecast = model.predict(future)
    
    # --- GRAFİK BİRLEŞTİRME ---
    fig = go.Figure()

    # 1. GEÇMİŞ VERİLER (Mavi Çizgi)
    # Son 6 ayı gösterelim ki grafik çok sıkışmasın (ama veri arkada var)
    gosterim_df = df.tail(180) 
    fig.add_trace(go.Scatter(
        x=gosterim_df.index, 
        y=gosterim_df['Close'], 
        mode='lines',
        name='Gerçekleşen Fiyat', 
        line=dict(color='#00BFFF', width=3) # Mavi
    ))

    # 2. GELECEK TAHMİNİ (Turuncu Kesik Çizgi)
    future_forecast = forecast.tail(14)
    fig.add_trace(go.Scatter(
        x=future_forecast['ds'], 
        y=future_forecast['yhat'], 
        mode='lines',
        name='Yapay Zeka Tahmini', 
        line=dict(color='#FFA500', width=3, dash='dot') # Turuncu ve Kesik
    ))

    # Grafik Ayarları
    fig.update_layout(
        title=f'{secilen_isim} - 6 Aylık Geçmiş ve 14 Günlük Tahmin',
        xaxis_title='Tarih',
        yaxis_title='Fiyat (TL)',
        template="plotly_dark",
        height=600,
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption("Not: Altın ve Gümüş fiyatları Ons/Dolar paritesi üzerinden Gram/TL olarak hesaplanmıştır.")

else:
    st.error("Veri çekilemedi. Lütfen sayfayı yenileyin veya sembolü kontrol edin.")