# -------------------------------------------------------------
#  📊 DASHBOARD FINANCIERO AVANZADO  –  ROIC GURÚFOCUS
# -------------------------------------------------------------
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

# -------------------------------------------------------------
# ⚙️ Configuración global de la página
# -------------------------------------------------------------
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# Parámetros por defecto (ajustables en el sidebar)
# -------------------------------------------------------------
Rf = 0.0435      # Risk-free rate
Rm = 0.085       # Expected market return
Tc = 0.21        # Tax rate (21 %)

# -------------------------------------------------------------
# ░░░░░   FUNCIONES AUXILIARES NUEVAS
# -------------------------------------------------------------
def safe_first(obj):
    """Devuelve el primer valor no nulo de una serie o None."""
    if obj is None:
        return None
    if hasattr(obj, "dropna") and hasattr(obj, "iloc"):
        s = obj.dropna()
        return s.iloc[0] if not s.empty else None
    return obj

def extraer_cash_equivalents(bs, info):
    keys = [
        "Cash And Cash Equivalents",
        "Cash And Cash Equivalents At Carrying Value",
        "Cash Cash Equivalents And Short Term Investments",
    ]
    for k in keys:
        if k in bs.index:
            return bs.loc[k]
    return pd.Series([info.get("totalCash")], index=bs.columns[:1])

def extraer_ebit(stock):
    posibles = ["EBIT", "Operating Income", "Earnings Before Interest and Taxes"]
    for k in posibles:
        if k in stock.financials.index:
            return stock.financials.loc[k]
        if k in stock.income_stmt.index:
            return stock.income_stmt.loc[k]
    return pd.Series([stock.info.get("ebit")], index=stock.financials.columns[:1])

def calcular_wacc(info, total_debt, Tc):
    beta  = info.get("beta", 1.0)
    price = info.get("currentPrice")
    shares = info.get("sharesOutstanding")
    market_cap = price * shares if price and shares else 0
    Re = Rf + beta * (Rm - Rf)
    Rd = 0.055 if total_debt else 0
    if market_cap + total_debt == 0:
        return None
    return ((market_cap / (market_cap + total_debt)) * Re +
            (total_debt  / (market_cap + total_debt)) * Rd * (1 - Tc))

def capital_invertido_prom(debt, equity, cash_eq):
    """Promedio 2 años de (Deuda + Equity – Efectivo)."""
    def ic(i):
        return (debt.iloc[i] or 0) + (equity.iloc[i] or 0) - (cash_eq.iloc[i] or 0)
    actual = ic(0)
    previo = ic(1) if len(debt) > 1 else actual
    return (actual + previo) / 2 or None

def calcular_roic_eva(stock, info, bs, Tc):
    ebit = extraer_ebit(stock)
    # Serie de deuda total (preferente)
    debt = bs.loc["Total Debt"] if "Total Debt" in bs.index else \
           (bs.loc.get("Long Term Debt", 0) + bs.loc.get("Short Term Debt", 0))
    equity = bs.loc["Total Stockholder Equity"] if "Total Stockholder Equity" in bs.index else \
             pd.Series([info.get("totalStockholderEquity")], index=bs.columns[:1])
    cash_eq = extraer_cash_equivalents(bs, info)

    inv_cap = capital_invertido_prom(debt, equity, cash_eq)
    nopat = safe_first(ebit)
    if nopat is not None:
        nopat *= (1 - Tc)

    roic = nopat / inv_cap if (nopat is not None and inv_cap) else None

    total_debt = safe_first(debt) or info.get("totalDebt") or 0
    wacc = calcular_wacc(info, total_debt, Tc)
    eva = (roic - wacc) * inv_cap if all(v is not None for v in (roic, wacc, inv_cap)) else None

    return roic, eva, wacc, inv_cap

def calcular_crecimiento_historico(fin, metric):
    if metric not in fin.index:
        return None
    vals = fin.loc[metric].dropna().iloc[:4]
    if len(vals) < 2 or vals.iloc[-1] == 0:
        return None
    return (vals.iloc[0] / vals.iloc[-1]) ** (1 / (len(vals)-1)) - 1

# -------------------------------------------------------------
# ░░░░░   OBTENER DATOS POR TICKER
# -------------------------------------------------------------
def obtener_datos_financieros(ticker):
    try:
        stock = yf.Ticker(ticker)
        info  = stock.info
        bs    = stock.balance_sheet
        fin   = stock.financials
        cf    = stock.cashflow

        # ---- ROIC, EVA, WACC (nuevo método) ---------------------------
        roic, eva, wacc, inv_cap = calcular_roic_eva(stock, info, bs, Tc)

        # ---- Otros indicadores ----------------------------------------
        price   = info.get("currentPrice")
        name    = info.get("longName", ticker)
        sector  = info.get("sector", "N/D")

        pe      = info.get("trailingPE")
        pb      = info.get("priceToBook")
        divid_y = info.get("dividendYield")
        payout  = info.get("payoutRatio")
        roe     = info.get("returnOnEquity")
        roa     = info.get("returnOnAssets")
        curr_ra = info.get("currentRatio")
        quick_ra= info.get("quickRatio")
        de      = info.get("debtToEquity")
        ltde    = info.get("longTermDebtToEquity")
        op_marg = info.get("operatingMargins")
        pr_marg = info.get("profitMargins")

        fcf = cf.loc["Free Cash Flow"].iloc[0] if "Free Cash Flow" in cf.index else None
        shares = info.get("sharesOutstanding")
        pfcf = price / (fcf / shares) if (fcf and shares) else None

        rev_g = calcular_crecimiento_historico(fin, "Total Revenue")
        eps_g = calcular_crecimiento_historico(fin, "Net Income")
        fcf_g = calcular_crecimiento_historico(cf, "Free Cash Flow") \
                or calcular_crecimiento_historico(cf, "Operating Cash Flow")

        return {
            "Ticker": ticker,
            "Nombre": name,
            "Sector": sector,
            "Precio": price,
            "P/E": pe,
            "P/B": pb,
            "P/FCF": pfcf,
            "Dividend Yield %": divid_y,
            "Payout Ratio": payout,
            "ROA": roa,
            "ROE": roe,
            "Current Ratio": curr_ra,
            "Quick Ratio": quick_ra,
            "Debt/Eq": de,
            "LtDebt/Eq": ltde,
            "Oper Margin": op_marg,
            "Profit Margin": pr_marg,
            "WACC": wacc,
            "ROIC": roic,
            "EVA": eva,
            "Revenue Growth": rev_g,
            "EPS Growth": eps_g,
            "FCF Growth": fcf_g,
        }
    except Exception as e:
        return {"Ticker": ticker, "Error": str(e)}

# -------------------------------------------------------------
# ░░░░░   INTERFAZ STREAMLIT
# -------------------------------------------------------------
def main():
    st.title("📊 Dashboard de Análisis Financiero Avanzado")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        tickers_in = st.text_area("🔎 Tickers (coma)", "HRL, AAPL, MSFT")
        max_t = st.slider("Máx tickers", 1, 100, 50)
        st.markdown("---")
        global Rf, Rm, Tc
        Rf = st.number_input("Risk-free (%)", 0.0, 20.0, 4.35) / 100
        Rm = st.number_input("Retorno mercado (%)", 0.0, 30.0, 8.5) / 100
        Tc = st.number_input("Tax rate (%)", 0.0, 50.0, 21.0) / 100

    tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()][:max_t]

    if st.button("🔍 Analizar", type="primary"):
        if not tickers:
            st.warning("Ingresa al menos un ticker")
            return

        res, pb = {}, st.progress(0)
        for i, t in enumerate(tickers, 1):
            res[t] = obtener_datos_financieros(t)
            pb.progress(i / len(tickers))
            time.sleep(1)
        pb.empty()

        df = pd.DataFrame([d for d in res.values() if "Error" not in d])
        if df.empty:
            st.error("Sin datos válidos")
            return

        # ----- formateo porcentual -----
        pct_cols = ["Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                    "Oper Margin", "Profit Margin", "WACC", "ROIC"]
        for c in pct_cols:
            if c in df.columns:
                df[c] = df[c].apply(lambda x: f"{x*100:,.2f}%" if pd.notnull(x) else "N/D")

        # =========================================================
        st.header("📋 Resumen General")
        show_cols = ["Ticker", "Sector", "Precio", "P/E", "P/B", "P/FCF",
                     "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                     "Current Ratio", "Debt/Eq", "Oper Margin", "Profit Margin",
                     "WACC", "ROIC", "EVA"]
        st.dataframe(df[show_cols].dropna(how="all", axis=1),
                     use_container_width=True, height=400)

        # -----------------------------------------------------
        # Sección 2 - Análisis de Valoración
        # -----------------------------------------------------
        st.header("💰 Análisis de Valoración")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ratios de Valoración")
            fig, ax = plt.subplots(figsize=(10, 4))
            df_plot = df[["Ticker", "P/E", "P/B", "P/FCF"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
            df_plot.plot(kind="bar", ax=ax, rot=45)
            ax.set_title("Comparativa de Ratios de Valoración")
            ax.set_ylabel("Ratio")
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Dividendos")
            fig, ax = plt.subplots(figsize=(10, 4))
            df_plot = df[["Ticker", "Dividend Yield %"]].set_index("Ticker")
            df_plot["Dividend Yield %"] = df_plot["Dividend Yield %"].replace("N/D", 0)
            df_plot["Dividend Yield %"] = df_plot["Dividend Yield %"].str.rstrip("%").astype("float")
            df_plot.plot(kind="bar", ax=ax, rot=45)
            ax.set_title("Rendimiento de Dividendos (%)")
            ax.set_ylabel("Dividend Yield %")
            st.pyplot(fig)
            plt.close()

        # -----------------------------------------------------
        # Sección 3 - Rentabilidad y Eficiencia
        # -----------------------------------------------------
        st.header("📈 Rentabilidad y Eficiencia")
        tabs = st.tabs(["ROE vs ROA", "Márgenes", "WACC vs ROIC"])

        with tabs[0]:
            fig, ax = plt.subplots(figsize=(10, 5))
            df_plot = df[["Ticker", "ROE", "ROA"]].set_index("Ticker")
            df_plot["ROE"] = df_plot["ROE"].str.rstrip("%").astype("float")
            df_plot["ROA"] = df_plot["ROA"].str.rstrip("%").astype("float")
            df_plot.plot(kind="bar", ax=ax, rot=45)
            ax.set_title("ROE vs ROA (%)")
            ax.set_ylabel("Porcentaje")
            st.pyplot(fig)
            plt.close()

        with tabs[1]:
            fig, ax = plt.subplots(figsize=(10, 5))
            df_plot = df[["Ticker", "Oper Margin", "Profit Margin"]].set_index("Ticker")
            df_plot["Oper Margin"] = df_plot["Oper Margin"].str.rstrip("%").astype("float")
            df_plot["Profit Margin"] = df_plot["Profit Margin"].str.rstrip("%").astype("float")
            df_plot.plot(kind="bar", ax=ax, rot=45)
            ax.set_title("Margen Operativo vs Margen Neto (%)")
            ax.set_ylabel("Porcentaje")
            st.pyplot(fig)
            plt.close()

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for _, row in df.iterrows():
                wacc_val = float(row["WACC"].rstrip("%")) if row["WACC"] != "N/D" else None
                roic_val = float(row["ROIC"].rstrip("%")) if row["ROIC"] != "N/D" else None
                if wacc_val is not None and roic_val is not None:
                    color = "green" if roic_val > wacc_val else "red"
                    ax.bar(row["Ticker"], roic_val, color=color, alpha=0.6, label="ROIC")
                    ax.bar(row["Ticker"], wacc_val, color="gray", alpha=0.3, label="WACC")
            ax.set_title("Creación de Valor: ROIC vs WACC (%)")
            ax.set_ylabel("Porcentaje")
            ax.legend()
            st.pyplot(fig)
            plt.close()

        # -----------------------------------------------------
        # Sección 4 - Análisis de Deuda
        # -----------------------------------------------------
        st.header("🏦 Estructura de Capital y Deuda")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Apalancamiento")
            fig, ax = plt.subplots(figsize=(10, 5))
            df_plot = df[["Ticker", "Debt/Eq", "LtDebt/Eq"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
            df_plot.plot(kind="bar", stacked=True, ax=ax, rot=45)
            ax.axhline(1, color="red", linestyle="--")
            ax.set_title("Deuda/Patrimonio")
            ax.set_ylabel("Ratio")
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Liquidez")
            fig, ax = plt.subplots(figsize=(10, 5))
            df_plot = df[["Ticker", "Current Ratio", "Quick Ratio", "Cash Ratio"]].set_index("Ticker").apply(pd.to_numeric, errors="coerce")
            df_plot.plot(kind="bar", ax=ax, rot=45)
            ax.axhline(1, color="green", linestyle="--")
            ax.set_title("Ratios de Liquidez")
            ax.set_ylabel("Ratio")
            st.pyplot(fig)
            plt.close()

        # -----------------------------------------------------
        # Sección 5 - Crecimiento
        # -----------------------------------------------------
        st.header("🚀 Crecimiento Histórico")
        growth_metrics = ["Revenue Growth", "EPS Growth", "FCF Growth"]
        df_growth = df[["Ticker"] + growth_metrics].set_index("Ticker") * 100  # a %
        fig, ax = plt.subplots(figsize=(12, 6))
        df_growth.plot(kind="bar", ax=ax, rot=45)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Tasas de Crecimiento Anual (%)")
        ax.set_ylabel("Crecimiento %")
        st.pyplot(fig)
        plt.close()

        # -----------------------------------------------------
        # Sección 6 - Análisis Individual
        # -----------------------------------------------------
        st.header("🔍 Análisis por Empresa")
        selected_ticker = st.selectbox("Selecciona una empresa", df["Ticker"].unique())
        empresa = df[df["Ticker"] == selected_ticker].iloc[0]

        st.subheader(f"Análisis Detallado: {empresa['Nombre']}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Precio", f"${empresa['Precio']:,.2f}" if empresa['Precio'] else "N/D")
            st.metric("P/E", empresa['P/E'])
            st.metric("P/B", empresa['P/B'])
        with col2:
            st.metric("ROE", empresa['ROE'])
            st.metric("ROIC", empresa['ROIC'])
            st.metric("WACC", empresa['WACC'])
        with col3:
            st.metric("Deuda/Patrimonio", empresa['Debt/Eq'])
            st.metric("Margen Neto", empresa['Profit Margin'])
            st.metric("Dividend Yield", empresa['Dividend Yield %'])

        st.subheader("Creación de Valor")
        fig, ax = plt.subplots(figsize=(6, 4))
        if empresa['ROIC'] != "N/D" and empresa['WACC'] != "N/D":
            roic_val = float(empresa['ROIC'].rstrip("%"))
            wacc_val = float(empresa['WACC'].rstrip("%"))
            color = "green" if roic_val > wacc_val else "red"
            ax.bar(["ROIC", "WACC"], [roic_val, wacc_val], color=[color, "gray"])
            ax.set_title("Creación de Valor (ROIC vs WACC)")
            ax.set_ylabel("%")
            st.pyplot(fig)
            plt.close()
            if roic_val > wacc_val:
                st.success("✅ La empresa está creando valor (ROIC > WACC)")
            else:
                st.error("❌ La empresa está destruyendo valor (ROIC < WACC)")
        else:
            st.warning("Datos insuficientes para análisis ROIC/WACC")

# -------------------------------------------------------------
# Punto de entrada
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
