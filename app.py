import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# --- AYARLAR ---
st.set_page_config(page_title="Finans Takip & Tahmin", layout="wide")

# Takip edilecek semboller
semboller = {
    'USD/TRY (Dolar)': 'USDTRY=X',
    'EUR/TRY (Euro)': 'EURTRY=X',
    'GBP/TRY (Sterlin)': 'GBPTRY=X',
    'Gram Altın (Tahmini)': 'GC=F', 
    'Gümüş (Ons)': 'SI=F',
    'AZN/TRY (Azerbaycan Manatı)': 'AZNTRY=X',
    'SAR/TRY (Suudi Riyali)': 'SARTRY=X',
    'KWD/TRY (Kuveyt Dinarı)': 'KWDTRY=X',
    'JPY/TRY (Japon Yeni)': 'JPYTRY=X',
}

# --- YAN MENÜ ---
st.sidebar.title("💰 Finans Paneli")
secilen_isim = st.sidebar.selectbox("Hangi Birimi İncelemek İstersin?", list(semboller.keys()))
secilen_sembol = semboller[secilen_isim]

tarih_araligi = st.sidebar.selectbox(
    "Zaman Aralığı",
    ("1 Günlük", "1 Haftalık", "1 Aylık", "3 Aylık", "1 Yıllık", "5 Yıllık")
)

# Zaman haritası
periyot_map = {
    "1 Günlük": "1d", "1 Haftalık": "5d", "1 Aylık": "1mo",
    "3 Aylık": "3mo", "1 Yıllık": "1y", "5 Yıllık": "5y"
}
aralik_map = {
    "1 Günlük": "5m", "1 Haftalık": "30m", "1 Aylık": "1h",
    "3 Aylık": "1d", "1 Yıllık": "1d", "5 Yıllık": "1wk"
}

# --- VERİ ÇEKME FONKSİYONU ---
@st.cache_data(ttl=60)
def veri_getir(sembol, periyot, aralik):
    # Veriyi indirirken multi-level index sorununu çözmek için auto_adjust=True kullanabiliriz
    # veya veriyi aldıktan sonra işleyebiliriz.
    df = yf.download(tickers=sembol, period=periyot, interval=aralik, progress=False)
    
    # Sütun isimleri bazen karmaşık (MultiIndex) gelebilir, düzeltelim:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Sadece 'Close', 'Open' gibi ana başlıkları al
            df.columns = df.columns.get_level_values(0)
        except:
            pass
            
    # Gram altın hesabı
    if sembol == 'GC=F': 
        usd_data = yf.download('USDTRY=X', period='1d', interval='1m', progress=False)
        if isinstance(usd_data.columns, pd.MultiIndex):
            usd_data.columns = usd_data.columns.get_level_values(0)
            
        usd_try = float(usd_data['Close'].iloc[-1])
        
        # Hesaplama
        df['Close'] = (df['Close'] * usd_try) / 31.1035
        df['Open'] = (df['Open'] * usd_try) / 31.1035
        df['High'] = (df['High'] * usd_try) / 31.1035
        df['Low'] = (df['Low'] * usd_try) / 31.1035
        
    return df

# --- ANA EKRAN ---
st.title(f"📈 {secilen_isim} Analizi")

try:
    df = veri_getir(secilen_sembol, periyot_map[tarih_araligi], aralik_map[tarih_araligi])
    
    if not df.empty and 'Close' in df.columns:
        # HATA ÇÖZÜMÜ BURADA: Gelen veriyi float() ile kesin olarak sayıya çeviriyoruz.
        son_fiyat = float(df['Close'].iloc[-1])
        onceki_fiyat = float(df['Close'].iloc[0])
        degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Anlık Fiyat", value=f"{son_fiyat:.2f} ₺", delta=f"%{degisim:.2f}")
        col2.info(f"Son Güncelleme: {datetime.now().strftime('%H:%M')}")
        
        # --- GRAFİK ---
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Piyasa'))

        fig.update_layout(
            title=f'{secilen_isim} Fiyat Grafiği',
            yaxis_title='Fiyat (TL)',
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # --- TAHMİN BÖLÜMÜ ---
        st.divider()
        st.subheader("🤖 Yapay Zeka Tahmini (Beta)")
        if st.button("Gelecek Analizi Yap"):
             st.success(f"Analiz Tamamlandı: {secilen_isim} için trend **{'YÜKSELİŞ' if degisim > 0 else 'DÜŞÜŞ'}** yönünde görünüyor.")
             st.caption("Detaylı Prophet analizi bir sonraki güncellemede eklenecektir.")

        # Tablo
        with st.expander("Detaylı Veri Tablosu"):
            st.dataframe(df.sort_index(ascending=False).style.format("{:.2f}"))

    else:
        st.warning("Veri yüklenirken bir sorun oluştu veya piyasa şu an kapalı.")

except Exception as e:
    st.error(f"Beklenmedik bir hata: {e}")