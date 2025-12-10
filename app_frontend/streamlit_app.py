# app_frontend/streamlit_app.py
import streamlit as st
import requests
from PIL import Image
import numpy as np
import os

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Iris Classifier", layout="centered")

st.title("--Iris Flower Classifier--")
st.write("Insira as medidas e obtenha a predição da espécie e a certeza (probabilidades).")

# Show image
img_path = "app_frontend/assets/iris.jpg"
if os.path.exists(img_path):
    st.image(Image.open(img_path), caption="Iris", use_column_width=True)
else:
    st.info("Coloque uma imagem da flor em 'app_frontend/assets/iris.jpg' para visualizar aqui.")

st.sidebar.header("Parâmetros da flor (cm)")
sepal_length = st.sidebar.number_input("Comprimento da sépala (sepal_length)", min_value=0.0, value=5.1, step=0.1, format="%.2f")
sepal_width  = st.sidebar.number_input("Largura da sépala (sepal_width)", min_value=0.0, value=3.5, step=0.1, format="%.2f")
petal_length = st.sidebar.number_input("Comprimento da pétala (petal_length)", min_value=0.0, value=1.4, step=0.1, format="%.2f")
petal_width  = st.sidebar.number_input("Largura da pétala (petal_width)", min_value=0.0, value=0.2, step=0.1, format="%.2f")

if st.button("Prever"):
    payload = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    try:
        with st.spinner("Consultando o modelo..."):
            res = requests.post(API_URL, json=payload, timeout=10)
        if res.status_code != 200:
            st.error(f"Erro na API: {res.status_code} - {res.text}")
        else:
            data = res.json()
            pred = data["predicted_class"]
            confidence = float(data["confidence"])
            probs = data["probabilities"]  # [setosa, versicolor, virginica]

            st.success(f"Predição: **{pred}**")
            st.write(f"Grau de certeza: **{confidence*100:.2f}%**")

            # Mostrar barras com as probabilidades
            st.subheader("Probabilidades por classe")
            labels = ["Setosa", "Versicolor", "Virgínica"]
            # Implementa um DataFrame para exibir
            import pandas as pd
            df = pd.DataFrame({"classe": labels, "probabilidade": probs})
            df = df.set_index("classe")
            st.bar_chart(df)

            # Opcional: mostrar probabilidades de todas as classes
            st.write("Probabilidades (raw):", {labels[i]: f"{probs[i]*100:.2f}%" for i in range(len(labels))})
    except requests.exceptions.RequestException as e:
        st.error(f"Falha ao conectar com a API: {e}")
