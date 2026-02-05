import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Finans Takip v2.0", layout="wide")

# --- LİSTE VE AYARLAR ---
# Kaynak yönetimi: direct (Direkt) veya calc (Hesaplamalı)
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

# --- YAN MENÜ ---
st.sidebar.title("💰 Kur Seçimi")
secilen_isim = st.sidebar.selectbox("Para Birimi Seçiniz", list(varliklar.keys()))
secim_bilgisi = varliklar[secilen_isim]

# --- VERİ ÇEKME VE HESAPLAMA ---
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

# --- ARAYÜZ ---
st.title(f"📈 {secilen_isim}")

with st.spinner('Piyasa verileri alınıyor...'):
    df = veri_getir(secim_bilgisi)

if not df.empty and len(df) > 1:
    son_fiyat = float(df['Close'].iloc[-1])
    onceki_fiyat = float(df['Close'].iloc[-2])
    degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(label="Anlık Değer", value=f"{son_fiyat:.2f} ₺", delta=f"%{degisim:.2f}")
    
    # --- YAPAY ZEKA VE GRAFİK ---
    try:
        # Prophet Hazırlığı
        df_prophet = df.reset_index()[['Date', 'Close']]
        df_prophet.columns = ['ds', 'y']
        
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=14)
        forecast = model.predict(future)
        
        # Çizim Ayarları (Türkçeleştirme burada)
        fig = go.Figure()
        gosterim_df = df.tail(180)
        
        # Geçmiş Veri Çizgisi
        fig.add_trace(go.Scatter(
            x=gosterim_df.index, 
            y=gosterim_df['Close'], 
            mode='lines', 
            name='Gerçekleşen Fiyat', 
            line=dict(color='#00BFFF', width=3),
            hovertemplate='<b>Tarih</b>: %{x|%d.%m.%Y}<br><b>Fiyat</b>: %{y:.2f} TL<extra></extra>' # Türkçe Tooltip
        ))
        
        # Tahmin Çizgisi
        future_forecast = forecast.tail(14)
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'], 
            y=future_forecast['yhat'], 
            mode='lines', 
            name='Yapay Zeka Tahmini', 
            line=dict(color='#FFA500', width=3, dash='dot'),
            hovertemplate='<b>Tarih</b>: %{x|%d.%m.%Y}<br><b>Tahmin</b>: %{y:.2f} TL<extra></extra>' # Türkçe Tooltip
        ))
        
        fig.update_layout(
            title=f'{secilen_isim} - Analiz ve Tahmin',
            xaxis_title='Tarih',
            yaxis_title='Değer (TL)',
            template="plotly_dark",
            height=550,
            hovermode="x unified", # Mouse ile gezince tüm bilgileri göster
            legend=dict(orientation="h", y=1.02, x=0, title=None)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Alt Bilgi Notu
        st.info("💡 **Bilgi:** Grafikteki turuncu kesik çizgiler, yapay zekanın geçmiş verilere dayanarak öngördüğü muhtemel trenddir. Kesinlik içermez.")

    except Exception as e:
        st.line_chart(df['Close'])

else:
    st.error("⚠️ Veri şu anda alınamıyor. Piyasalar kapalı olabilir veya veri sağlayıcıda yoğunluk var.")