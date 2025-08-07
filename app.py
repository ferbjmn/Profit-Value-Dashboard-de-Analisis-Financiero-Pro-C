# ──────────────────────────────────────────────────────────────
#  📊  DASHBOARD FINANCIERO AVANZADO
#      ROIC / EVA estilo GuruFocus + manejo robusto de errores
# ──────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import time

# -------------------------------------------------------------
# ⚙️ Configuración global
# -------------------------------------------------------------
st.set_page_config(
    page_title="📊 Dashboard Financiero Avanzado",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="expanded",
)

# Parámetros por defecto (editables)
Rf, Rm, Tc = 0.0435, 0.085, 0.21

# -------------------------------------------------------------
# ░░░░░  Funciones auxiliares
# -------------------------------------------------------------
def safe_first(obj):
    if obj is None:
        return None
    if hasattr(obj, "dropna"):
        obj = obj.dropna()
        return obj.iloc[0] if not obj.empty else None
    return obj

def get_cash_equiv(bs, info):
    keys = ["Cash And Cash Equivalents",
            "Cash And Cash Equivalents At Carrying Value",
            "Cash Cash Equivalents And Short Term Investments"]
    for k in keys:
        if k in bs.index:
            return bs.loc[k]
    return pd.Series([info.get("totalCash")], index=bs.columns[:1])

def get_ebit(tkr):
    for k in ["EBIT", "Operating Income", "Earnings Before Interest and Taxes"]:
        if k in tkr.financials.index:
            return tkr.financials.loc[k]
        if k in tkr.income_stmt.index:
            return tkr.income_stmt.loc[k]
    return pd.Series([tkr.info.get("ebit")], index=tkr.financials.columns[:1])

def invested_cap_avg(debt, equity, cash_eq):
    ic0 = (debt.iloc[0] or 0)+(equity.iloc[0] or 0)-(cash_eq.iloc[0] or 0)
    ic1 = (debt.iloc[1] or 0)+(equity.iloc[1] or 0)-(cash_eq.iloc[1] or 0) if len(debt) > 1 else ic0
    return (ic0 + ic1) / 2 or None

def wacc(info, total_debt, Tc):
    beta  = info.get("beta", 1.0)
    price = info.get("currentPrice")
    shares = info.get("sharesOutstanding")
    mcap = price * shares if price and shares else 0
    Re, Rd = Rf + beta*(Rm-Rf), 0.055 if total_debt else 0
    if mcap + total_debt == 0:
        return None
    return (mcap/(mcap+total_debt))*Re + (total_debt/(mcap+total_debt))*Rd*(1-Tc)

def calc_cagr(df, metric):
    if metric not in df.index:
        return None
    vals = df.loc[metric].dropna().iloc[:4]
    if len(vals) < 2 or vals.iloc[-1] == 0:
        return None
    return (vals.iloc[0] / vals.iloc[-1]) ** (1/(len(vals)-1)) - 1

def calc_roic_eva(tkr, info, bs, Tc):
    ebit   = get_ebit(tkr)
    debt   = bs.loc["Total Debt"] if "Total Debt" in bs.index else \
             (bs.loc.get("Long Term Debt", 0) + bs.loc.get("Short Term Debt", 0))
    equity = bs.loc["Total Stockholder Equity"] if "Total Stockholder Equity" in bs.index else \
             pd.Series([info.get("totalStockholderEquity")], index=bs.columns[:1])
    cash_eq = get_cash_equiv(bs, info)
    inv_cap = invested_cap_avg(debt, equity, cash_eq)
    nopat = safe_first(ebit)
    if nopat is not None:
        nopat *= (1 - Tc)
    roic = nopat / inv_cap if (nopat is not None and inv_cap) else None
    wacc_val = wacc(info, safe_first(debt) or info.get("totalDebt") or 0, Tc)
    eva = (roic - wacc_val) * inv_cap if all(v is not None for v in (roic, wacc_val, inv_cap)) else None
    return roic, eva, wacc_val

# -------------------------------------------------------------
# ░░░░░  Descarga y procesa un ticker (con reintento)
# -------------------------------------------------------------
def fetch_ticker(tkr, Tc, retries=2):
    for attempt in range(retries):
        try:
            stk  = yf.Ticker(tkr, threads=False)
            info = stk.info
            bs   = stk.balance_sheet
            if not info or bs.empty:
                raise ValueError("info o balance_sheet vacío")
            fin  = stk.financials
            cf   = stk.cashflow

            roic, eva, wacc_val = calc_roic_eva(stk, info, bs, Tc)

            fcf = cf.loc["Free Cash Flow"].iloc[0] if "Free Cash Flow" in cf.index else None
            shr = info.get("sharesOutstanding")
            pfcf = info.get("currentPrice") / (fcf/shr) if (fcf and shr) else None

            return {
                "Ticker": tkr,
                "Sector": info.get("sector"),
                "Precio": info.get("currentPrice"),
                "P/E": info.get("trailingPE"),
                "P/B": info.get("priceToBook"),
                "P/FCF": pfcf,
                "Dividend Yield %": info.get("dividendYield"),
                "Payout Ratio": info.get("payoutRatio"),
                "ROA": info.get("returnOnAssets"),
                "ROE": info.get("returnOnEquity"),
                "Current Ratio": info.get("currentRatio"),
                "Quick Ratio": info.get("quickRatio"),
                "Debt/Eq": info.get("debtToEquity"),
                "LtDebt/Eq": info.get("longTermDebtToEquity"),
                "Oper Margin": info.get("operatingMargins"),
                "Profit Margin": info.get("profitMargins"),
                "WACC": wacc_val,
                "ROIC": roic,
                "EVA": eva,
                "Revenue Growth": calc_cagr(fin, "Total Revenue"),
                "EPS Growth":     calc_cagr(fin, "Net Income"),
                "FCF Growth":     calc_cagr(cf, "Free Cash Flow") or calc_cagr(cf, "Operating Cash Flow"),
            }
        except Exception as e:
            if attempt == retries - 1:
                return {"Ticker": tkr, "Error": str(e)}
            time.sleep(2)

# -------------------------------------------------------------
# ░░░░░  Streamlit UI
# -------------------------------------------------------------
def main():
    st.title("📊 Dashboard Financiero Avanzado")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        tickers_in = st.text_area("Tickers (separados por coma)", "HRL, AAPL, MSFT, GOOGL")
        max_t = st.slider("Máx tickers", 1, 100, 50)
        st.markdown("---")
        global Rf, Rm, Tc
        Rf = st.number_input("Risk-free rate (%)", 0.0, 20.0, 4.35) / 100
        Rm = st.number_input("Market return (%)", 0.0, 30.0, 8.5) / 100
        Tc = st.number_input("Tax rate (%)", 0.0, 50.0, 21.0) / 100

    tickers = [t.strip().upper() for t in tickers_in.split(",") if t.strip()][:max_t]

    if st.button("🔍 Analizar", type="primary"):
        if not tickers:
            st.warning("Ingresa al menos un ticker");  return

        results, pb = {}, st.progress(0)
        for i, tk in enumerate(tickers, 1):
            results[tk] = fetch_ticker(tk, Tc)
            pb.progress(i / len(tickers))
        pb.empty()

        df_ok  = pd.DataFrame([v for v in results.values() if "Error" not in v])
        df_err = pd.DataFrame([v for v in results.values() if "Error" in v])

        if df_ok.empty:
            st.error("No se obtuvo información válida para ningún ticker.")
            if not df_err.empty:
                st.subheader("🚫 Errores")
                st.table(df_err)
            return

        # Formateo de %
        pct_cols = ["Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                    "Oper Margin", "Profit Margin", "WACC", "ROIC"]
        for c in pct_cols:
            if c in df_ok.columns:
                df_ok[c] = df_ok[c].apply(lambda x: f"{x*100:,.2f}%" if pd.notnull(x) else "N/D")

        # -----------------------------------------------------
        # 1. Resumen General
        # -----------------------------------------------------
        st.header("📋 Resumen General")
        cols = ["Ticker", "Sector", "Precio", "P/E", "P/B", "P/FCF",
                "Dividend Yield %", "Payout Ratio", "ROA", "ROE",
                "Current Ratio", "Debt/Eq", "Oper Margin", "Profit Margin",
                "WACC", "ROIC", "EVA"]
        st.dataframe(df_ok[cols].dropna(how="all", axis=1),
                     use_container_width=True, height=380)

        if not df_err.empty:
            st.subheader("🚫 Tickers con error")
            st.table(df_err)

        # -----------------------------------------------------
        # 2. Análisis de Valoración
        # -----------------------------------------------------
        st.header("💰 Análisis de Valoración")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ratios de Valoración")
            fig, ax = plt.subplots(figsize=(9, 4))
            df_ok[["Ticker", "P/E", "P/B", "P/FCF"]].set_index("Ticker")\
                 .apply(pd.to_numeric, errors="coerce").plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("Ratio")
            st.pyplot(fig); plt.close()

        with col2:
            st.subheader("Dividend Yield (%)")
            fig, ax = plt.subplots(figsize=(9, 4))
            dy = df_ok[["Ticker", "Dividend Yield %"]].replace("N/D", 0)
            dy["Dividend Yield %"] = dy["Dividend Yield %"].str.rstrip("%").astype(float)
            dy.set_index("Ticker").plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            st.pyplot(fig); plt.close()

        # -----------------------------------------------------
        # 3. Rentabilidad y Eficiencia
        # -----------------------------------------------------
        st.header("📈 Rentabilidad y Eficiencia")
        tabs = st.tabs(["ROE vs ROA", "Márgenes", "WACC vs ROIC"])

        with tabs[0]:
            fig, ax = plt.subplots(figsize=(10, 5))
            rr = df_ok[["Ticker", "ROE", "ROA"]].replace("N/D", 0)
            rr["ROE"] = rr["ROE"].str.rstrip("%").astype(float)
            rr["ROA"] = rr["ROA"].str.rstrip("%").astype(float)
            rr.set_index("Ticker").plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            st.pyplot(fig); plt.close()

        with tabs[1]:
            fig, ax = plt.subplots(figsize=(10, 5))
            mm = df_ok[["Ticker", "Oper Margin", "Profit Margin"]].replace("N/D", 0)
            mm["Oper Margin"] = mm["Oper Margin"].str.rstrip("%").astype(float)
            mm["Profit Margin"] = mm["Profit Margin"].str.rstrip("%").astype(float)
            mm.set_index("Ticker").plot(kind="bar", ax=ax, rot=45)
            ax.set_ylabel("%")
            st.pyplot(fig); plt.close()

        with tabs[2]:
            fig, ax = plt.subplots(figsize=(10, 5))
            for _, r in df_ok.iterrows():
                w = float(r["WACC"].rstrip("%")) if r["WACC"] != "N/D" else None
                rt = float(r["ROIC"].rstrip("%")) if r["ROIC"] != "N/D" else None
                if w is not None and rt is not None:
                    ax.bar(r["Ticker"], rt, color="green" if rt > w else "red", alpha=0.6)
                    ax.bar(r["Ticker"], w, color="gray", alpha=0.3)
            ax.set_ylabel("%")
            ax.set_title("Creación de Valor (ROIC vs WACC)")
            st.pyplot(fig); plt.close()

        # -----------------------------------------------------
        # 4. Deuda y Liquidez
        # -----------------------------------------------------
        st.header("🏦 Deuda & Liquidez")
        c3, c4 = st.columns(2)

        with c3:
            st.subheader("Apalancamiento")
            fig, ax = plt.subplots(figsize=(9, 4))
            df_ok[["Ticker", "Debt/Eq", "LtDebt/Eq"]].set_index("Ticker")\
                 .apply(pd.to_numeric, errors="coerce").plot(kind="bar", stacked=True, ax=ax, rot=45)
            ax.axhline(1, color="red", linestyle="--")
            st.pyplot(fig); plt.close()

        with c4:
            st.subheader("Liquidez")
            fig, ax = plt.subplots(figsize=(9, 4))
            df_ok[["Ticker", "Current Ratio", "Quick Ratio"]]\
                 .set_index("Ticker").apply(pd.to_numeric, errors="coerce")\
                 .plot(kind="bar", ax=ax, rot=45)
            ax.axhline(1, color="green", linestyle="--")
            st.pyplot(fig); plt.close()

        # -----------------------------------------------------
        # 5. Crecimiento
        # -----------------------------------------------------
        st.header("🚀 Crecimiento (CAGR 3-4 años)")
        growth = df_ok.set_index("Ticker")[["Revenue Growth", "EPS Growth", "FCF Growth"]] * 100
        fig, ax = plt.subplots(figsize=(12, 6))
        growth.plot(kind="bar", ax=ax, rot=45)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("%")
        st.pyplot(fig); plt.close()

        # -----------------------------------------------------
        # 6. Análisis Individual
        # -----------------------------------------------------
        st.header("🔍 Análisis Individual")
        pick = st.selectbox("Selecciona empresa", df_ok["Ticker"].unique())
        det = df_ok[df_ok["Ticker"] == pick].iloc[0]

        cA, cB, cC = st.columns(3)
        with cA:
            st.metric("Precio", f"${det['Precio']:,.2f}" if det['Precio'] else "N/D")
            st.metric("P/E", det["P/E"])
            st.metric("P/B", det["P/B"])
        with cB:
            st.metric("ROIC", det["ROIC"])
            st.metric("WACC", det["WACC"])
            st.metric("EVA", f"{det['EVA']:,.0f}" if pd.notnull(det['EVA']) else "N/D")
        with cC:
            st.metric("ROE", det["ROE"])
            st.metric("Dividend Yield", det["Dividend Yield %"])
            st.metric("Debt/Eq", det["Debt/Eq"])

        st.subheader("ROIC vs WACC")
        if det["ROIC"] != "N/D" and det["WACC"] != "N/D":
            rv = float(det["ROIC"].rstrip("%"))
            wv = float(det["WACC"].rstrip("%"))
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar(["ROIC", "WACC"], [rv, wv], color=["green" if rv > wv else "red", "gray"])
            ax.set_ylabel("%")
            st.pyplot(fig)
            if rv > wv:
                st.success("✅ Crea valor (ROIC > WACC)")
            else:
                st.error("❌ Destruye valor (ROIC < WACC)")
        else:
            st.info("Datos insuficientes para comparar ROIC/WACC")

# -------------------------------------------------------------
# 🏁 Ejecución
# -------------------------------------------------------------
if __name__ == "__main__":
    main()
