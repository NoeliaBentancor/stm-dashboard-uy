from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

DATA_PATH = Path("data/processed/stm_hourly_line.parquet")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DATA_PATH)
    df["timestamp_hour"] = pd.to_datetime(df["timestamp_hour"])
    df["fecha"] = df["timestamp_hour"].dt.date
    df["hora"] = df["timestamp_hour"].dt.hour
    day_names = {
        0: "Lunes",
        1: "Martes",
        2: "Miércoles",
        3: "Jueves",
        4: "Viernes",
        5: "Sábado",
        6: "Domingo",
    }
    df["dia_semana_num"] = df["timestamp_hour"].dt.dayofweek
    df["dia_semana"] = df["dia_semana_num"].map(day_names)
    return df


def build_forecast(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float] | None:
    ts = (
        df.groupby("timestamp_hour", as_index=False)["demanda"]
        .sum()
        .set_index("timestamp_hour")
        .asfreq("h", fill_value=0)
    )
    ts["hour"] = ts.index.hour
    ts["dow"] = ts.index.dayofweek
    ts["month"] = ts.index.month
    ts["lag_1"] = ts["demanda"].shift(1)
    ts["lag_24"] = ts["demanda"].shift(24)
    ts["lag_168"] = ts["demanda"].shift(168)
    ts["rolling_24"] = ts["demanda"].shift(1).rolling(24).mean()
    ts = ts.dropna()

    if len(ts) < 24 * 14:
        return None

    test_size = min(max(24 * 7, int(len(ts) * 0.2)), 24 * 21)
    train = ts.iloc[:-test_size]
    test = ts.iloc[-test_size:]

    feature_cols = ["hour", "dow", "month", "lag_1", "lag_24", "lag_168", "rolling_24"]
    model = RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(train[feature_cols], train["demanda"])
    pred = model.predict(test[feature_cols])

    result = test[["demanda"]].copy()
    result["predicho"] = pred
    result["residuo"] = result["demanda"] - result["predicho"]
    sigma = result["residuo"].std(ddof=0) or 1.0
    result["anomalia"] = result["residuo"].abs() > (2.5 * sigma)

    mae = mean_absolute_error(result["demanda"], result["predicho"])
    denominator = result["demanda"].replace(0, np.nan).abs()
    mape = ((result["demanda"] - result["predicho"]).abs() / denominator).mean() * 100
    mape = float(0 if np.isnan(mape) else mape)
    return result.reset_index(), float(mae), mape


st.set_page_config(page_title="STM Dashboard", layout="wide")
st.title("STM Montevideo · Demanda + Forecast + Anomalías")

df = load_data()
if df.empty:
    st.warning(
        "No hay datos procesados. Ejecutá:\n\n"
        "`python3 -m src.ingest --months 1`"
    )
    st.stop()

with st.sidebar:
    st.header("Filtros")
    empresas = sorted([str(x) for x in df["descrip_empresa"].dropna().unique().tolist()])
    empresa_sel = st.multiselect("Empresa", empresas, default=empresas)

    lineas = sorted([str(x) for x in df["dsc_linea"].dropna().unique().tolist()])
    linea_sel = st.multiselect("Línea", lineas, default=lineas[:15] if len(lineas) > 15 else lineas)

    min_date = pd.to_datetime(df["fecha"]).min().date()
    max_date = pd.to_datetime(df["fecha"]).max().date()
    date_range = st.date_input("Rango de fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)

filtered = df.copy()
if empresa_sel:
    filtered = filtered[filtered["descrip_empresa"].astype(str).isin(empresa_sel)]
if linea_sel:
    filtered = filtered[filtered["dsc_linea"].astype(str).isin(linea_sel)]
if isinstance(date_range, tuple) and len(date_range) == 2:
    filtered = filtered[(filtered["fecha"] >= date_range[0]) & (filtered["fecha"] <= date_range[1])]

if filtered.empty:
    st.warning("No hay datos para esos filtros.")
    st.stop()

total_demanda = int(filtered["demanda"].sum())
top_line = (
    filtered.groupby("dsc_linea", as_index=False)["demanda"].sum().sort_values("demanda", ascending=False).head(1)
)
top_hour = (
    filtered.groupby("hora", as_index=False)["demanda"].sum().sort_values("demanda", ascending=False).head(1)
)

col1, col2, col3 = st.columns(3)
col1.metric("Demanda total (ascensos)", f"{total_demanda:,}")
col2.metric("Línea #1", str(top_line.iloc[0]["dsc_linea"]) if not top_line.empty else "-")
col3.metric("Hora pico", f"{int(top_hour.iloc[0]['hora']):02d}:00" if not top_hour.empty else "-")

daily = filtered.groupby("fecha", as_index=False)["demanda"].sum()
fig_daily = px.line(daily, x="fecha", y="demanda", title="Demanda diaria")
st.plotly_chart(fig_daily, use_container_width=True)

top_lines = (
    filtered.groupby("dsc_linea", as_index=False)["demanda"].sum().sort_values("demanda", ascending=False).head(20)
)
fig_lines = px.bar(top_lines, x="dsc_linea", y="demanda", title="Top 20 líneas por demanda")
st.plotly_chart(fig_lines, use_container_width=True)

heat = (
    filtered.groupby(["dia_semana_num", "dia_semana", "hora"], as_index=False)["demanda"]
    .sum()
    .pivot(index="dia_semana", columns="hora", values="demanda")
)
desired_order = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]
heat = heat.reindex(desired_order)
fig_heat = px.imshow(heat, aspect="auto", title="Heatmap demanda · día x hora", color_continuous_scale="Viridis")
st.plotly_chart(fig_heat, use_container_width=True)

st.subheader("Forecast y detección de anomalías")
forecast_result = build_forecast(filtered)
if forecast_result is None:
    st.info("No hay suficientes horas para entrenar el modelo. Probá ampliar el rango o líneas.")
else:
    forecast_df, mae, mape = forecast_result
    c1, c2 = st.columns(2)
    c1.metric("MAE (test)", f"{mae:,.2f}")
    c2.metric("MAPE (test)", f"{mape:,.2f}%")

    fig_pred = go.Figure()
    fig_pred.add_trace(
        go.Scatter(
            x=forecast_df["timestamp_hour"],
            y=forecast_df["demanda"],
            mode="lines",
            name="Real",
        )
    )
    fig_pred.add_trace(
        go.Scatter(
            x=forecast_df["timestamp_hour"],
            y=forecast_df["predicho"],
            mode="lines",
            name="Predicho",
        )
    )
    anom = forecast_df[forecast_df["anomalia"]]
    fig_pred.add_trace(
        go.Scatter(
            x=anom["timestamp_hour"],
            y=anom["demanda"],
            mode="markers",
            name="Anomalía",
            marker=dict(size=8, color="red"),
        )
    )
    fig_pred.update_layout(title="Real vs predicho (ventana de test)")
    st.plotly_chart(fig_pred, use_container_width=True)

    st.dataframe(
        anom[["timestamp_hour", "demanda", "predicho", "residuo"]]
        .sort_values("timestamp_hour", ascending=False)
        .head(50),
        use_container_width=True,
    )

