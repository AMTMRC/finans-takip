import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="Finans Takip", layout="wide")

# --- LİSTE VE TİCKER EŞLEŞTİRMELERİ ---
# ARTIK HEPSİNİ DOLAR ÜZERİNDEN HESAPLAYACAĞIZ (GARANTİ YÖNTEM)
# 'source': 'direct' -> Direkt veriyi çek (USD, EUR, GBP gibi ana kurlar için)
# 'source': 'calc'   -> Dolar paritesi üzerinden hesapla (Verisi zor bulunanlar için)
varliklar = {
    'USD - Amerikan Doları': {'ticker': 'USDTRY=X', 'source': 'direct'},
    'EUR - Avrupa Para Birimi': {'ticker': 'EURTRY=X', 'source': 'direct'},
    'GBP - İngiliz Sterlini': {'ticker': 'GBPTRY=X', 'source': 'direct'},
    'XAU - Altın (Gram)': {'ticker': 'GC=F', 'source': 'gold_calc'},
    'XAG - Gümüş (Gram)': {'ticker': 'SI=F', 'source': 'silver_calc'},
    
    # --- ÇAPRAZ KUR İLE HESAPLANACAKLAR (Verisi Garanti Olanlar) ---
    'CAD - Kanada Doları': {'ticker': 'USDCAD=X', 'source': 'calc'},
    'CHF - İsviçre Frangı': {'ticker': 'CHF=X', 'source': 'calc_inverse'}, # USDCHF farklı yazılır
    'AUD - Avustralya Doları': {'ticker': 'AUDUSD=X', 'source': 'calc_multiply'}, # AUDUSD tersten yazılır
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

# --- AKILLI VERİ ÇEKME FONKSİYONU ---
@st.cache_data(ttl=300)
def veri_getir(info):
    try:
        # Önce her zaman Dolar/TL kurunu çekelim (Hesaplamalar için lazım)
        usd_try_df = yf.download('USDTRY=X', period="2y", interval="1d", progress=False)
        if isinstance(usd_try_df.columns, pd.MultiIndex): usd_try_df.columns = usd_try_df.columns.get_level_values(0)
        
        # Eğer Dolar/TL verisi yoksa hiç başlama
        if usd_try_df.empty: return pd.DataFrame()

        # 1. DİREKT MOD (USD, EUR, GBP)
        if info['source'] == 'direct':
            df = yf.download(info['ticker'], period="2y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df

        # 2. ALTIN/GÜMÜŞ MODU
        elif info['source'] in ['gold_calc', 'silver_calc']:
            df = yf.download(info['ticker'], period="2y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
            # Veri setlerini eşle
            df = df.reindex(usd_try_df.index).dropna()
            eslesmis_usd = usd_try_df.reindex(df.index)
            
            # Formül: (Ons * Dolar) / 31.1035
            for col in ['Close', 'Open', 'High', 'Low']:
                df[col] = (df[col] * eslesmis_usd['Close']) / 31.1035
            return df

        # 3. ÇAPRAZ KUR HESAPLAMA MODU (AZN, SEK, DKK vs.)
        else:
            # Hedef pariteyi çek (Örn: USD/AZN)
            target_df = yf.download(info['ticker'], period="2y", interval="1d", progress=False)
            if isinstance(target_df.columns, pd.MultiIndex): target_df.columns = target_df.columns.get_level_values(0)
            
            # Tarihleri eşle
            common_index = usd_try_df.index.intersection(target_df.index)
            usd_try_filtered = usd_try_df.loc[common_index]
            target_filtered = target_df.loc[common_index]
            
            df = pd.DataFrame(index=common_index)
            
            # HESAPLAMA TİPLERİ
            if info['source'] == 'calc': 
                # Örnek: AZN (Manat). 1 USD = 34 TL, 1 USD = 1.7 AZN. 
                # 1 AZN = 34 / 1.7 = 20 TL. Formül: USDTRY / USDAZN
                df['Close'] = usd_try_filtered['Close'] / target_filtered['Close']
                
            elif info['source'] == 'calc_inverse':
                # Örnek: CHF (İsviçre). Yahoo USDCHF değil CHF=X (Dolar endeksi gibi) verir bazen.
                # Genelde: USDTRY * USDCHF_Paritesi (Eğer kur ters ise)
                # Standart: USDTRY / USDCHF
                df['Close'] = usd_try_filtered['Close'] * target_filtered['Close'] # CHF genelde terstir

            elif info['source'] == 'calc_multiply':
                # Örnek: AUD (Avustralya). Ticker AUDUSD=X (1 AUD kaç USD).
                # 1 AUD = 0.65 USD. 1 USD = 34 TL.
                # 1 AUD = 0.65 * 34. Formül: USDTRY * AUDUSD
                df['Close'] = usd_try_filtered['Close'] * target_filtered['Close']

            return df

    except Exception as e:
        return pd.DataFrame()

# --- ARAYÜZ ---
st.title(f"📈 {secilen_isim}")

with st.spinner('Global piyasalar taranıyor ve TL karşılığı hesaplanıyor...'):
    df = veri_getir(secim_bilgisi)

if not df.empty and len(df) > 1:
    son_fiyat = float(df['Close'].iloc[-1])
    onceki_fiyat = float(df['Close'].iloc[-2])
    degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
    
    st.metric(label="Anlık Değer (TL)", value=f"{son_fiyat:.2f} ₺", delta=f"%{degisim:.2f}")
    
    # --- GRAFİK VE TAHMİN ---
    try:
        # Prophet Hazırlığı
        df_prophet = df.reset_index()[['Date', 'Close']]
        df_prophet.columns = ['ds', 'y']
        
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=14)
        forecast = model.predict(future)
        
        # Çizim
        fig = go.Figure()
        gosterim_df = df.tail(180)
        
        fig.add_trace(go.Scatter(x=gosterim_df.index, y=gosterim_df['Close'], mode='lines', name='Gerçekleşen', line=dict(color='#00BFFF', width=3)))
        
        future_forecast = forecast.tail(14)
        fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='YZ Tahmini', line=dict(color='#FFA500', width=3, dash='dot')))
        
        fig.update_layout(title=f'{secilen_isim} Analizi', yaxis_title='Fiyat (TL)', template="plotly_dark", height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.line_chart(df['Close'])

else:
    st.error("Veri hesaplanamadı. Piyasalar kapalı olabilir.")