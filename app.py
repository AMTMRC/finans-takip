import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Finans Takip", layout="wide")

# --- LİSTE ---
# Yahoo Finance'de en güvenilir veri kodlarını kullanıyoruz
semboller = {
    'USD - Amerikan Doları': 'USDTRY=X',
    'EUR - Avrupa Para Birimi': 'EURTRY=X',
    'GBP - İngiliz Sterlini': 'GBPTRY=X',
    'XAU - Altın (Gram)': 'GC=F',       
    'XAG - Gümüş (Gram)': 'SI=F',       
    'CAD - Kanada Doları': 'CADTRY=X',
    'CHF - İsviçre Frangı': 'CHFTRY=X',
    'AUD - Avustralya Doları': 'AUDTRY=X',
    'DKK - Danimarka Kronu': 'DKKTRY=X',
    'JPY - Japon Yeni': 'JPYTRY=X',
    'KWD - Kuveyt Dinarı': 'KWDTRY=X', # Bazen veri gelmeyebilir
    'NOK - Norveç Kronu': 'NOKTRY=X',
    'SAR - Suudi Arabistan Riyali': 'SARTRY=X',
    'SEK - İsveç Kronu': 'SEKTRY=X',
    'AED - Bae Dirhemi': 'AEDTRY=X',
    'AZN - Azerbaycan Manatı': 'AZNTRY=X', # Veri sorunu sık yaşanır
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
        # Veriyi çek
        df = yf.download(sembol, period="2y", interval="1d", progress=False)
        
        # Sütun düzeltme (MultiIndex sorununu çözer)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        # Boş veri kontrolü
        if df.empty:
            return pd.DataFrame()

        # Altın/Gümüş Hesaplaması (Ons -> Gram TL)
        if sembol in ['GC=F', 'SI=F']:
            usd_data = yf.download('USDTRY=X', period='1d', progress=False)
            if not usd_data.empty:
                if isinstance(usd_data.columns, pd.MultiIndex):
                    usd_data.columns = usd_data.columns.get_level_values(0)
                
                usd_try = float(usd_data['Close'].iloc[-1])
                # Hesaplama
                for col in ['Close', 'Open', 'High', 'Low']:
                    if col in df.columns:
                        df[col] = (df[col] * usd_try) / 31.1035

        return df
    except Exception as e:
        return pd.DataFrame() # Hata olursa boş tablo döndür

# --- ANA EKRAN ---
st.title(f"📈 {secilen_isim}")

with st.spinner('Piyasa verileri analiz ediliyor...'):
    df = veri_getir(secilen_sembol)

# --- VERİ KONTROLÜ VE GÖSTERİMİ ---
# Burası en önemli kısım: Veri var mı ve yeterli mi diye bakıyoruz
if not df.empty and 'Close' in df.columns and len(df) > 1:
    
    try:
        son_fiyat = float(df['Close'].iloc[-1])
        onceki_fiyat = float(df['Close'].iloc[-2])
        degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
        
        st.metric(label="Anlık Değer (TL)", value=f"{son_fiyat:.2f} ₺", delta=f"%{degisim:.2f}")
        
        # --- YAPAY ZEKA KISMI ---
        try:
            df_prophet = df.reset_index()[['Date', 'Close']]
            df_prophet.columns = ['ds', 'y']
            
            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)
            
            future = model.make_future_dataframe(periods=14)
            forecast = model.predict(future)
            
            # Grafik
            fig = go.Figure()
            
            # Geçmiş (Son 180 gün)
            gosterim_df = df.tail(180)
            fig.add_trace(go.Scatter(x=gosterim_df.index, y=gosterim_df['Close'], mode='lines', name='Gerçekleşen', line=dict(color='#00BFFF', width=3)))
            
            # Gelecek Tahmini
            future_forecast = forecast.tail(14)
            fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='YZ Tahmini', line=dict(color='#FFA500', width=3, dash='dot')))
            
            fig.update_layout(title=f'{secilen_isim} Analizi', yaxis_title='Fiyat (TL)', template="plotly_dark", height=600, hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning("Yapay zeka tahmini oluşturulurken küçük bir sorun oldu, ancak güncel veriler yukarıda.")
            st.line_chart(df['Close'])

    except IndexError:
        st.warning("Veri kaynağında anlık bir kopukluk var, lütfen sayfayı yenileyin.")

elif not df.empty and len(df) == 1:
    # Sadece tek bir fiyat verisi varsa (Geçmiş yoksa)
    son_fiyat = float(df['Close'].iloc[-1])
    st.metric(label="Anlık Değer (TL)", value=f"{son_fiyat:.2f} ₺")
    st.warning("Bu para birimi için yeterli geçmiş veri bulunamadı, sadece anlık fiyat gösteriliyor.")

else:
    # Hiç veri yoksa
    st.error(f"⚠️ '{secilen_isim}' için şu anda borsadan veri çekilemiyor.")
    st.info("Bunun sebebi piyasaların kapalı olması veya Yahoo Finance sunucularındaki geçici bir kesinti olabilir. Lütfen Dolar veya Euro gibi ana para birimlerini deneyin.")