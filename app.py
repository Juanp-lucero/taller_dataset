from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

INCOME = "Annual Income (k$)"
SPENDING = "Spending Score (1-100)"
REQUIRED = ["CustomerID", "Gender", "Age", INCOME, SPENDING]

st.set_page_config(page_title="Segmentación de clientes", page_icon="🛍️", layout="wide")
st.title("Segmentación de clientes con K-Means")
st.caption("Mall Customers · análisis de ingreso anual y puntuación de gasto")

@st.cache_data
def load_csv(file_bytes, file_name):
    from io import BytesIO
    data = pd.read_csv(BytesIO(file_bytes))
    missing = [col for col in REQUIRED if col not in data.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")
    return data

upload = st.sidebar.file_uploader("Carga el CSV de clientes", type=["csv"])
if upload is not None:
    try:
        data = load_csv(upload.getvalue(), upload.name)
    except Exception as exc:
        st.error(f"No se pudo leer el CSV: {exc}")
        st.stop()
else:
    default_path = Path(__file__).with_name("Mall_Customers.csv")
    if not default_path.exists():
        st.info("Carga Mall_Customers.csv en la barra lateral para comenzar.")
        st.stop()
    data = pd.read_csv(default_path)

features = [INCOME, SPENDING]
if st.sidebar.checkbox("Incluir edad en la segmentación", value=False):
    features = ["Age"] + features
k = st.sidebar.slider("Número de clusters (K)", min_value=2, max_value=10, value=5)

scaler = StandardScaler()
X = scaler.fit_transform(data[features])
model = KMeans(n_clusters=k, random_state=42, n_init=20)
data = data.copy()
data["Cluster"] = model.fit_predict(X)
centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=features)

main_tab, clusters_tab, classify_tab = st.tabs(["Evaluación de K", "Clusters y perfiles", "Clasificar cliente"])
with main_tab:
    metrics = []
    for candidate in range(2, 11):
        candidate_model = KMeans(n_clusters=candidate, random_state=42, n_init=20)
        candidate_labels = candidate_model.fit_predict(X)
        metrics.append((candidate, candidate_model.inertia_, silhouette_score(X, candidate_labels)))
    metric_df = pd.DataFrame(metrics, columns=["K", "Inercia", "Silueta"])
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots()
        ax.plot(metric_df["K"], metric_df["Inercia"], marker="o")
        ax.set(xlabel="K", ylabel="Inercia", title="Método del codo")
        ax.grid(alpha=.25)
        st.pyplot(fig)
        plt.close(fig)
    with c2:
        fig, ax = plt.subplots()
        ax.plot(metric_df["K"], metric_df["Silueta"], marker="o", color="#d95f02")
        ax.axvline(k, linestyle="--", color="gray", label=f"K seleccionado: {k}")
        ax.set(xlabel="K", ylabel="Silueta", title="Coeficiente de silueta")
        ax.grid(alpha=.25)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    st.dataframe(metric_df.round(4), width="stretch", hide_index=True)

with clusters_tab:
    left, right = st.columns([2, 1])
    with left:
        fig, ax = plt.subplots(figsize=(8, 5))
        for cluster_id, group in data.groupby("Cluster"):
            ax.scatter(group[INCOME], group[SPENDING], label=f"Cluster {cluster_id}", alpha=.8)
        ax.scatter(centers[INCOME], centers[SPENDING], marker="X", s=220, c="black", label="Centroides")
        ax.set(xlabel=INCOME, ylabel=SPENDING, title=f"Segmentos encontrados (K={k})")
        ax.legend()
        ax.grid(alpha=.2)
        st.pyplot(fig)
        plt.close(fig)
    summary = data.groupby("Cluster").agg(
        Clientes=("CustomerID", "count"), Edad_media=("Age", "mean"),
        Ingreso_medio=(INCOME, "mean"), Gasto_medio=(SPENDING, "mean")
    ).round(1)
    with right:
        st.subheader("Resumen por grupo")
        st.dataframe(summary, width="stretch")
        st.caption("Los números de cluster son etiquetas, no un ranking de valor.")
    st.subheader("Datos con cluster asignado")
    st.dataframe(data, width="stretch", hide_index=True)
    st.download_button("Descargar clientes segmentados", data.to_csv(index=False).encode("utf-8"), file_name="clientes_segmentados.csv", mime="text/csv")

with classify_tab:
    st.write("Ingresa características en la escala original del dataset.")
    with st.form("new_customer"):
        age = st.number_input("Edad", min_value=18, max_value=100, value=30)
        income = st.number_input("Ingreso anual (k$)", min_value=0.0, max_value=300.0, value=60.0, step=1.0)
        spending = st.slider("Puntuación de gasto (1–100)", min_value=1, max_value=100, value=50)
        submitted = st.form_submit_button("Clasificar cliente")
    if submitted:
        new_row = {"Age": age, INCOME: income, SPENDING: spending}
        new_X = pd.DataFrame([[new_row[col] for col in features]], columns=features)
        cluster = int(model.predict(scaler.transform(new_X))[0])
        st.success(f"El cliente queda asignado al Cluster {cluster}.")
        st.write("Centroide del grupo (escala original):")
        st.dataframe(centers.iloc[[cluster]].round(2), width="stretch")

st.sidebar.caption("Modelo demostrativo con datos de clientes de centro comercial; no interpreta causalidad ni usa datos transaccionales reales.")
