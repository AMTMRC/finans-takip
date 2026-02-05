import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Finans Asistanı Pro", layout="wide", initial_sidebar_state="expanded")

# --- CSS (Görsel İyileştirme) ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #0e1117; border-radius: 5px; color: white; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #262730; color: #FFA500; }
    .stMetric { background-color: #0e1117; border: 1px solid #333; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- VARLIK LİSTESİ (Çapraz Kur Destekli) ---
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

# --- MERKEZİ VERİ FONKSİYONU ---
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
st.sidebar.title("Finans Asistanı 🚀")
st.sidebar.markdown("---")
secilen_isim = st.sidebar.selectbox("Ana Para Birimi", list(varliklar.keys()))
st.sidebar.info("Uygulama üstündeki sekmelerden (Tab) mod değiştirebilirsiniz.")

# --- ANA VERİYİ ÇEK ---
ana_df = veri_getir(varliklar[secilen_isim])

# --- SEKMELER ---
tab1, tab2, tab3 = st.tabs(["🔮 Analiz & Tahmin", "🏎️ Karşılaştırma", "⏳ Zaman Makinesi"])

# ---------------------------------------------------------
# TAB 1: MEVCUT SİSTEM (ANALİZ & TAHMİN)
# ---------------------------------------------------------
with tab1:
    st.header(f"📈 {secilen_isim} Analizi")
    
    if not ana_df.empty and len(ana_df) > 1:
        son_fiyat = float(ana_df['Close'].iloc[-1])
        onceki_fiyat = float(ana_df['Close'].iloc[-2])
        degisim = ((son_fiyat - onceki_fiyat) / onceki_fiyat) * 100
        
        c1, c2 = st.columns([1, 4])
        with c1:
            st.metric("Anlık Değer", f"{son_fiyat:.2f} ₺", f"%{degisim:.2f}")
        
        # Prophet
        try:
            df_prophet = ana_df.reset_index()[['Date', 'Close']]
            df_prophet.columns = ['ds', 'y']
            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)
            future = model.make_future_dataframe(periods=14)
            forecast = model.predict(future)
            
            fig = go.Figure()
            gosterim_df = ana_df.tail(180)
            
            fig.add_trace(go.Scatter(x=gosterim_df.index, y=gosterim_df['Close'], mode='lines', name='Gerçekleşen', line=dict(color='#00BFFF', width=3), hovertemplate='%{y:.2f} TL'))
            
            future_forecast = forecast.tail(14)
            fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], mode='lines', name='YZ Tahmini', line=dict(color='#FFA500', width=3, dash='dot'), hovertemplate='%{y:.2f} TL'))
            
            fig.update_layout(template="plotly_dark", height=500, hovermode="x unified", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
            
            # Excel İndirme
            csv = ana_df.to_csv().encode('utf-8')
            st.download_button("📥 Verileri Excel Olarak İndir", csv, f"{secilen_isim}_veri.csv", "text/csv")
            
        except Exception:
            st.line_chart(ana_df['Close'])
    else:
        st.error("Veri alınamadı.")

# ---------------------------------------------------------
# TAB 2: KARŞILAŞTIRMA MODU
# ---------------------------------------------------------
with tab2:
    st.header("🆚 Varlık Yarışı")
    st.write("Seçtiğin para birimlerinin son 1 yıldaki performansını karşılaştır.")
    
    karsilastirma_listesi = st.multiselect("Karşılaştırılacakları Seçin:", list(varliklar.keys()), default=['USD - Amerikan Doları', 'EUR - Avrupa Para Birimi', 'XAU - Altın (Gram)'])
    
    if karsilastirma_listesi:
        fig_comp = go.Figure()
        
        with st.spinner('Veriler karşılaştırılıyor...'):
            for varlik in karsilastirma_listesi:
                df_temp = veri_getir(varliklar[varlik])
                if not df_temp.empty:
                    # Son 1 yılı al
                    df_temp = df_temp.tail(365)
                    # Normalize et (Yüzdesel değişim için: İlk gün 100 kabul edilir)
                    ilk_fiyat = float(df_temp['Close'].iloc[0])
                    df_temp['Normalize'] = (df_temp['Close'] / ilk_fiyat) * 100
                    
                    fig_comp.add_trace(go.Scatter(
                        x=df_temp.index, 
                        y=df_temp['Normalize'], 
                        mode='lines', 
                        name=varlik.split(' - ')[0], # Sadece kısa kodu göster
                        hovertemplate='%{y:.1f} Puan'
                    ))
        
        fig_comp.update_layout(
            title="Son 1 Yıl Performans Karşılaştırması (Başlangıç=100)",
            yaxis_title="Büyüme Endeksi",
            template="plotly_dark",
            height=600,
            hovermode="x unified"
        )
        st.plotly_chart(fig_comp, use_container_width=True)
        st.caption("Not: Grafikteki çizgiler 'fiyatı' değil 'kazandırma oranını' gösterir. Çizgisi en üstte olan en çok kazandırandır.")

# ---------------------------------------------------------
# TAB 3: ZAMAN MAKİNESİ (SİMÜLATÖR)
# ---------------------------------------------------------
with tab3:
    st.header("⏳ Yatırım Simülatörü")
    st.write("Geçmişte yatırım yapsaydın bugün ne kadar paran olurdu?")
    
    col1, col2, col3 = st.columns(3)
    tutar = col1.number_input("Yatırılan Tutar (TL)", value=10000, step=1000)
    tarih = col2.date_input("Hangi Tarihte?", value=datetime.now() - timedelta(days=365))
    varlik_secim = col3.selectbox("Hangi Varlık?", list(varliklar.keys()))
    
    hesapla_btn = st.button("💸 Hesapla")
    
    if hesapla_btn:
        df_sim = veri_getir(varliklar[varlik_secim])
        secilen_tarih_str = tarih.strftime('%Y-%m-%d')
        
        if not df_sim.empty:
            # Seçilen tarihe en yakın veriyi bul
            try:
                gecmis_veri = df_sim.iloc[df_sim.index.get_indexer([secilen_tarih_str], method='nearest')]
                gecmis_fiyat = float(gecmis_veri['Close'].values[0])
                guncel_fiyat = float(df_sim['Close'].iloc[-1])
                
                # Hesaplama
                adet = tutar / gecmis_fiyat
                guncel_deger = adet * guncel_fiyat
                kar_zarar = guncel_deger - tutar
                yuzde = (kar_zarar / tutar) * 100
                
                st.markdown("---")
                c1, c2, c3 = st.columns(3)
                c1.metric("O Gün Alınan Miktar", f"{adet:.2f} Adet")
                c2.metric("Bugünkü Değeri", f"{guncel_deger:,.2f} TL", f"%{yuzde:.1f}")
                c3.metric("Net Kar/Zarar", f"{kar_zarar:,.2f} TL")
                
                if kar_zarar > 0:
                    st.success(f"🎉 Tebrikler! Eğer o gün alıp bekleseydin paran **{yuzde:.1f} kat** artacaktı.")
                else:
                    st.error("📉 Maalesef, bu yatırım o tarihten beri değer kaybetmiş.")
                    
            except Exception as e:
                st.error("Seçilen tarih için veri bulunamadı. Lütfen daha yakın bir tarih seçin.")