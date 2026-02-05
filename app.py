import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import datetime, timedelta

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Finans Asistanı", layout="wide", initial_sidebar_state="expanded")

# --- CSS VE STİL ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { height: 45px; background-color: #0e1117; border-radius: 5px; color: white; font-size: 14px; }
    .stTabs [data-baseweb="tab"]:hover { background-color: #262730; color: #FFA500; }
    .stMetric { background-color: #1f2937; border: 1px solid #374151; padding: 10px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# --- VARLIK LİSTESİ (SADECE İSTEDİKLERİN) ---
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

# --- CÜZDAN SİSTEMİ ---
if 'portfoy' not in st.session_state:
    st.session_state.portfoy = []

# --- FONKSİYONLAR ---
def rsi_hesapla(series, period=14):
    delta = series.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

@st.cache_data(ttl=300)
def veri_getir(info):
    try:
        # Temel Dolar Verisi (Hesaplamalar İçin)
        usd_try_df = yf.download('USDTRY=X', period="2y", interval="1d", progress=False)
        if isinstance(usd_try_df.columns, pd.MultiIndex): usd_try_df.columns = usd_try_df.columns.get_level_values(0)
        
        if usd_try_df.empty: return pd.DataFrame()

        # 1. Direkt Veriler
        if info['source'] == 'direct':
            df = yf.download(info['ticker'], period="2y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            return df

        # 2. Altın/Gümüş Hesaplaması
        elif info['source'] in ['gold_calc', 'silver_calc']:
            df = yf.download(info['ticker'], period="2y", interval="1d", progress=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df = df.reindex(usd_try_df.index).dropna()
            eslesmis_usd = usd_try_df.reindex(df.index)
            for col in ['Close', 'Open', 'High', 'Low']:
                df[col] = (df[col] * eslesmis_usd['Close']) / 31.1035
            return df

        # 3. Çapraz Kur Hesaplamaları
        else:
            target_df = yf.download(info['ticker'], period="2y", interval="1d", progress=False)
            if isinstance(target_df.columns, pd.MultiIndex): target_df.columns = target_df.columns.get_level_values(0)
            common_index = usd_try_df.index.intersection(target_df.index)
            usd_try = usd_try_df.loc[common_index]
            target = target_df.loc[common_index]
            df = pd.DataFrame(index=common_index)
            
            if info['source'] == 'calc': df['Close'] = usd_try['Close'] / target['Close']
            elif info['source'] == 'calc_inverse': df['Close'] = usd_try['Close'] * target['Close']
            elif info['source'] == 'calc_multiply': df['Close'] = usd_try['Close'] * target['Close']
            return df
    except:
        return pd.DataFrame()

# --- ARAYÜZ ---
st.sidebar.title("Finans Asistanı 🚀")
secilen_isim = st.sidebar.selectbox("Varlık Seçimi", list(varliklar.keys()))

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Grafik Ayarları")
goster_sma50 = st.sidebar.checkbox("50 Günlük Ort.", value=False)
goster_rsi = st.sidebar.checkbox("RSI Göstergesi", value=True)

ana_df = veri_getir(varliklar[secilen_isim])

# --- SEKMELER ---
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Analiz", "💼 Cüzdanım", "🏎️ Yarış", "⏳ Simülatör"])

# TAB 1: ANALİZ
with tab1:
    if not ana_df.empty:
        son_fiyat = ana_df['Close'].iloc[-1]
        degisim = ((son_fiyat - ana_df['Close'].iloc[-2]) / ana_df['Close'].iloc[-2]) * 100
        
        c1, c2 = st.columns([1, 3])
        c1.metric("Fiyat", f"{son_fiyat:,.2f} ₺", f"%{degisim:.2f}")
        
        try:
            df_p = ana_df.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})
            m = Prophet(daily_seasonality=True).fit(df_p)
            forecast = m.predict(m.make_future_dataframe(periods=14))
            
            rows = 2 if goster_rsi else 1
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3] if goster_rsi else [1.0], vertical_spacing=0.05)
            
            fig.add_trace(go.Scatter(x=ana_df.index, y=ana_df['Close'], name='Fiyat', line=dict(color='#00BFFF')), row=1, col=1)
            fig.add_trace(go.Scatter(x=forecast['ds'].tail(14), y=forecast['yhat'].tail(14), name='Tahmin', line=dict(color='orange', dash='dot')), row=1, col=1)
            
            if goster_sma50:
                fig.add_trace(go.Scatter(x=ana_df.index, y=ana_df['Close'].rolling(50).mean(), name='SMA50', line=dict(color='yellow', width=1)), row=1, col=1)
                
            if goster_rsi:
                rsi = rsi_hesapla(ana_df['Close'])
                fig.add_trace(go.Scatter(x=ana_df.index, y=rsi, name='RSI', line=dict(color='purple')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
                
            fig.update_layout(template="plotly_dark", height=600, margin=dict(l=10,r=10,t=30,b=10), hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception:
            st.line_chart(ana_df['Close'])
    else:
        st.warning("Veri yükleniyor...")

# TAB 2: CÜZDAN
with tab2:
    st.header("💼 Varlıklarım")
    col1, col2, col3 = st.columns([2, 1, 1])
    p_varlik = col1.selectbox("Varlık Ekle", list(varliklar.keys()), key="p_select")
    p_adet = col2.number_input("Adet", min_value=0.01, value=1.0, step=1.0)
    
    if col3.button("➕ Ekle"):
        st.session_state.portfoy.append({"isim": p_varlik, "adet": p_adet})
        st.success("Eklendi!")

    st.divider()

    if st.session_state.portfoy:
        toplam_tl = 0
        df_pie = []
        for item in st.session_state.portfoy:
            df_temp = veri_getir(varliklar[item['isim']])
            if not df_temp.empty:
                tutar = item['adet'] * df_temp['Close'].iloc[-1]
                toplam_tl += tutar
                df_pie.append({"Varlık": item['isim'].split(' - ')[0], "Tutar": tutar})
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("Toplam Servet", f"{toplam_tl:,.2f} ₺")
            if st.button("🗑️ Temizle"):
                st.session_state.portfoy = []
                st.rerun()
            for item in st.session_state.portfoy:
                st.text(f"- {item['adet']} x {item['isim']}")
        with c2:
            if df_pie:
                fig_pie = px.pie(pd.DataFrame(df_pie), values='Tutar', names='Varlık', title='Dağılım', template="plotly_dark", hole=0.4)
                st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Cüzdan boş.")

# TAB 3: YARIŞ
with tab3:
    st.header("🏎️ Karşılaştırma")
    secilenler = st.multiselect("Varlıklar", list(varliklar.keys()), default=['USD - Amerikan Doları', 'EUR - Avrupa Para Birimi'])
    if secilenler:
        fig_race = go.Figure()
        for v in secilenler:
            d = veri_getir(varliklar[v])
            if not d.empty:
                d = d.tail(365)
                norm = (d['Close'] / d['Close'].iloc[0]) * 100
                fig_race.add_trace(go.Scatter(x=d.index, y=norm, mode='lines', name=v.split(' - ')[0]))
        fig_race.update_layout(template="plotly_dark", hovermode="x unified")
        st.plotly_chart(fig_race, use_container_width=True)

# TAB 4: SİMÜLATÖR
with tab4:
    st.header("⏳ Simülatör")
    c1, c2, c3 = st.columns(3)
    s_tutar = c1.number_input("Tutar (TL)", value=1000)
    s_tarih = c2.date_input("Tarih", value=datetime.now()-timedelta(days=365))
    s_varlik = c3.selectbox("Varlık", list(varliklar.keys()), key="sim_select")
    
    if st.button("Hesapla"):
        df_sim = veri_getir(varliklar[s_varlik])
        tarih_str = s_tarih.strftime('%Y-%m-%d')
        if not df_sim.empty:
            try:
                eski = df_sim.iloc[df_sim.index.get_indexer([tarih_str], method='nearest')]
                eski_fiyat = float(eski['Close'].values[0])
                yeni_fiyat = float(df_sim['Close'].iloc[-1])
                yeni_deger = (s_tutar / eski_fiyat) * yeni_fiyat
                st.success(f"Bugünkü Değer: **{yeni_deger:,.2f} TL**")
            except:
                st.error("Veri yok.")