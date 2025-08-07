# ──────────────────────────────────────────────────────────────
#  📊 DASHBOARD FINANCIERO AVANZADO
#     · ROIC y EVA ajustados a la metodología GuruFocus
# ──────────────────────────────────────────────────────────────
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
# Parámetros por defecto (editables en el sidebar)
# -------------------------------------------------------------
Rf = 0.0435      # Risk-free rate
Rm = 0.085       # Expected market return
Tc = 0.21        # Tax rate (21 %)

# -------------------------------------------------------------
# ░░░░░   FUNCIONES AUXILIARES
# -------------------------------------------------------------
def safe_first(obj):
    """Primer valor no nulo (o None) de Series/escalares."""
    if obj is None:
        return None
    if hasattr(obj, "dropna"):
        s = obj.dropna()
        return s.iloc[0] if not s.empty else None
    return obj

def extraer_cash_equivalents(bs, info):
    keys = ["Cash And Cash Equivalents",
            "Cash And Cash Equivalents At Carrying Value",
            "Cash Cash Equivalents And Short Term Investments"]
    for k in keys:
        if k in bs.index:
            return bs.loc[k]
    return pd.Series([info.get("totalCash")], index=bs.columns[:1])

def extraer_ebit(stock):
    opciones = ["EBIT", "Operating Income",
                "Earnings Before Interest and Taxes"]
    for k in opciones:
        if k in stock.financials.index:
            return stock.financials.loc[k]
        if k in stock.income_stmt.index:
            return stock.income_stmt.loc[k]
    return pd.Series([stock.info.get("ebit")], index=stock.financials.columns[:1])

def capital_invertido_prom(debt, equity, cash_eq):
    """Promedio 2 años de (Deuda + Equity − Cash)."""
    def ic(i):
        return (debt.iloc[i] or 0) + (equity.iloc[i] or 0) - (cash_eq.iloc[i] or 0)
    actual = ic(0)
    previo = ic(1) if len(debt) > 1 else actual
    return (actual + previo) / 2 or None

def calcular_wacc(info, total_debt, Tc):
    beta = info.get("beta", 1.0)
    price, shares = info.get("currentPrice"), info.get("sharesOutstanding")
    market_cap = price * shares if price and shares else 0
    Re = Rf + beta * (Rm - Rf)
    Rd = 0.055 if total_debt else 0
    if market_cap + total_debt == 0:
        return None
    return ((market_cap / (market_cap + total_debt)) * Re +
            (total_debt  / (market_cap + total_debt)) * Rd * (1 - Tc))

def calcular_roic_eva(stock, info, bs, Tc):
    """Devuelve (ROIC, EVA, WACC, Capital invertido)."""
    ebit = extraer_ebit(stock)
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

def cagr_4p(df, metric):
    if metric not in df.index:
        return None
    vals = df.loc[metric].dropna().iloc[:4]
    if len(vals) < 2 or vals.iloc[-1] == 0:
        return None
    return (vals.iloc[0] / vals.iloc[-1]) ** (1 / (len(vals)-1)) - 1

# -------------------------------------------------------------
# ░░░░░   DESCARGA Y PROCESO POR TICKER
# -------------------------------------------------------------
def obtener_datos_financieros(ticker, Tc):
    try:
        s = yf.Ticker(ticker)
        info, bs, fin, cf = s.info, s.balance_sheet, s.financials, s.cashflow

        roic, eva, wacc, inv_cap = calcular_roic_eva(s, info, bs, Tc)

        price   = info.get("currentPrice")
        sector  = info.get("sector", "N/D")
        pe      = info.get("trailingPE")
        pb      = info.get("priceToBook")
        dy      = info.get("dividendYield")
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

        rev_g = cagr_4p(fin, "Total Revenue")
        eps_g = cagr_4p(fin, "Net Income")
        fcf_g = cagr_4p(cf, "Free Cash Flow") or cagr_4p(cf, "Operating Cash Flow")

        return {
            "Ticker": ticker,
            "Sector": sector,
            "Precio": price,
            "P/E": pe,
            "P/B": pb,
            "P/FCF": pfcf,
            "Dividend Yield %": dy,
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

    # ---- Sidebar --------------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuración")
        tickers_in = st.text_area("🔎 Ingresa tickers (separados por coma)",
                                  "HRL, AAPL, MSFT, GOOGL, AMZN")
        max_tickers = st.slider("Número máximo de tickers", 1, 100, 50)
        st.markdown("---")
        global Rf, Rm, Tc
        Rf = st.number_input("Tasa libre de riesgo (%)", 0.0, 20.0, 4.35) / 100
        Rm = st.number_input("Retorno esperado del mercado (%)", 0.0, 30.0, 8.5) / 100
        Tc = st.number_input("Tasa impositiva corporativa (%)", 0.0, 50.0, 21.0) / 100

    tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()][:max_tickers]

    if st.button("🔍 Analizar Acciones", type="primary"):
        if not tickers:
            st.warning("Debes ingresar al menos un ticker")
            return

        resultados, pb = {}, st.progress(0)
        for i, t in enumerate(tickers, 1):
            resultados[t] = obtener_datos_financieros(t, Tc)
            pb.progress(i / len(tickers))
            time.sleep(1)
        pb.empty()

        datos = [d for d in resultados.values() if "Error" not in d]
        if not datos:
            st.error("No se obtuvo información válida.")
            return

        df = pd.DataFrame(datos)

        # ---- Formateo de porcentajes ------------------------------------
        pct_cols = ["Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                    "Oper Margin", "Profit Margin", "WACC", "ROIC"]
        for c in pct_cols:
            if c in df.columns:
                df[c] = df[c].apply(lambda x: f"{x*100:,.2f}%" if pd.notnull(x) else "N/D")

        # ================================================================
        # 1. Resumen General
        # ================================================================
        st.header("📋 Resumen General")
        cols = ["Ticker", "Sector", "Precio", "P/E", "P/B", "P/FCF",
                "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                "Current Ratio", "Debt/Eq", "Oper Margin", "Profit Margin",
                "WACC", "ROIC", "EVA"]
        st.dataframe(df[cols].dropna(how="all", axis=1),
                     use_container_width=True, height=380)

        # ================================================================
        # 2. Análisis de Valoración
        # ================================================================
        st.header("💰 Análisis de Valoración")
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Ratios de Valoración")
            fig, ax = plt.subplots(figsize=(9, 4))
            df_plot = df[["Ticker", "P/E", "P/B", "P/FCF"]]\
                        .set_index("Ticker").apply(pd.to_numeric, errors="coerce")
            df_plot.plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("Ratio")
            st.pyplot(fig)
            plt.close()

        with c2:
            st.subheader("Dividend Yield")
            fig, ax = plt.subplots(figsize=(9, 4))
            dy = df[["Ticker", "Dividend Yield %"]].replace("N/D", 0)
            dy["Dividend Yield %"] = dy["Dividend Yield %"].str.rstrip("%").astype(float)
            dy.set_index("Ticker").plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            st.pyplot(fig)
            plt.close()

        # ================================================================
        # 3. Rentabilidad y Eficiencia
        # ================================================================
        st.header("📈 Rentabilidad y Eficiencia")
        tabs = st.tabs(["ROE vs ROA", "Márgenes", "WACC vs ROIC"])

        with tabs[0]:
            fig, ax = plt.subplots(figsize=(10, 5))
            rr = df[["Ticker", "ROE", "ROA"]].replace("N/D", 0)
            rr["ROE"] = rr["ROE"].str.rstrip("%").astype(float)
            rr["ROA"] = rr["ROA"].str.rstrip("%").astype(float)
            rr.set_index("Ticker").plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            st.pyplot(fig)
            plt.close()

        with tabs[1]:
            fig, ax = plt.subplots(figsize=(10, 5))
            mm = df[["Ticker", "Oper Margin", "Profit Margin"]].replace("N/D", 0)
            mm["Oper Margin"] = mm["Oper Margin"].str.rstrip("%").astype(float)
            mm["Profit Margin"] = mm["Profit Margin"].str.rstrip("%").astype(float)
            mm.set_index("Ticker").plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            st.pyplot(fig)
            plt.close()

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for _, r in df.iterrows():
                wacc_v = float(r["WACC"].rstrip("%")) if r["WACC"] != "N/D" else None
                roic_v = float(r["ROIC"].rstrip("%")) if r["ROIC"] != "N/D" else None
                if wacc_v is not None and roic_v is not None:
                    col = "green" if roic_v > wacc_v else "red"
                    ax.bar(r["Ticker"], roic_v, color=col, alpha=0.6)
                    ax.bar(r["Ticker"], wacc_v, color="gray", alpha=0.3)
            ax.set_ylabel("%")
            ax.set_title("Creación de Valor: ROIC vs WACC")
            st.pyplot(fig)
            plt.close()

        # ================================================================
        # 4. Deuda y Liquidez
        # ================================================================
        st.header("🏦 Estructura de Capital y Deuda")
        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Apalancamiento")
            fig, ax = plt.subplots(figsize=(9, 4))
            de_plot = df[["Ticker", "Debt/Eq", "LtDebt/Eq"]]\
                        .set_index("Ticker").apply(pd.to_numeric, errors="coerce")
            de_plot.plot(kind="bar", stacked=True, ax=ax, rot=45)
            ax.axhline(1, color="red", linestyle="--")
            st.pyplot(fig)
            plt.close()

        with c4:
            st.subheader("Liquidez")
            fig, ax = plt.subplots(figsize=(9, 4))
            liq = df[["Ticker", "Current Ratio", "Quick Ratio"]]\
                     .set_index("Ticker").apply(pd.to_numeric, errors="coerce")
            liq.plot(kind="bar", ax=ax, rot=45)
            ax.axhline(1, color="green", linestyle="--")
            st.pyplot(fig)
            plt.close()

        # ================================================================
        # 5. Crecimiento Histórico
        # ================================================================
        st.header("🚀 Crecimiento Histórico (CAGR 3-4 años)")
        growth = df.set_index("Ticker")[["Revenue Growth", "EPS Growth", "FCF Growth"]]*100
        fig, ax = plt.subplots(figsize=(12, 6))
        growth.plot(kind="bar", ax=ax, rot=45)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("%")
        st.pyplot(fig)
        plt.close()

        # ================================================================
        # 6. Análisis Individual
        # ================================================================
        st.header("🔍 Análisis Individual")
        pick = st.selectbox("Elige empresa", df["Ticker"].unique())
        det = df[df["Ticker"] == pick].iloc[0]

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Precio", f"${det['Precio']:,.2f}" if det['Precio'] else "N/D")
            st.metric("P/E", det["P/E"])
            st.metric("P/B", det["P/B"])
        with colB:
            st.metric("ROIC", det["ROIC"])
            st.metric("WACC", det["WACC"])
            st.metric("EVA", f"{det['EVA']:,.0f}" if pd.notnull(det['EVA']) else "N/D")
        with colC:
            st.metric("ROE", det["ROE"])
            st.metric("Dividend Yield", det["Dividend Yield %"])
            st.metric("Debt/Eq", det["Debt/Eq"])

        st.subheader("ROIC vs WACC")
        if det["ROIC"] != "N/D" and det["WACC"] != "N/D":
            r_val = float(det["ROIC"].rstrip("%"))
            w_val = float(det["WACC"].rstrip("%"))
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(["ROIC", "WACC"], [r_val, w_val],
                   color=["green" if r_val > w_val else "red", "gray"])
            ax.set_ylabel("%")
            st.pyplot(fig)
            if r_val > w_val:
                st.success("✅ La empresa está creando valor (ROIC > WACC)")
            else:
                st.error("❌ La empresa destruye valor (ROIC < WACC)")
        else:
            st.info("Datos insuficientes para comparar ROIC/WACC")

# -------------------------------------------------------------
# 🏁 EJECUCIÓN
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
