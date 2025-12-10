import streamlit as st
import requests
from PIL import Image
import numpy as np
import os

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Penguim Classifier", layout="centered")

st.title("--Penguim Species Classifier--")
st.write("Insira as medidas e obtenha a predição da espécie e a certeza (probabilidades).")

# Show image
img_path = "app_frontend/assets/penguim.jpg"
if os.path.exists(img_path):
    st.image(Image.open(img_path), caption="Iris", use_column_width=True)
else:
    st.info("Coloque uma imagem do penguim em 'app_frontend/assets/iris.jpg' para visualizar aqui.")


# DEFINIÇÃO DOS PARAMETROS DE ENTRADA
st.sidebar.header("Parâmetros dos penguins:")
flipper_length_mm = st.sidebar.number_input("Comprimento da nadadeira em mm (flipper_length_mm)", min_value=0.0, value=5.1, step=0.1, format="%.2f")
body_mass_g  = st.sidebar.number_input("Massa corporal em g (body_mass_g)", min_value=0.0, value=3.5, step=0.1, format="%.2f")
culmen_length_mm = st.sidebar.number_input("Comprimento da crista superior do bico em mm (culmen_length_mm)", min_value=0.0, value=1.4, step=0.1, format="%.2f")
culmen_depth_mm  = st.sidebar.number_input("Profundidade da crista superior do bico em mm (culmen_depth_mm)", min_value=0.0, value=0.2, step=0.1, format="%.2f")


if st.button("Prever"):
    payload = {
        "sepal_length": flipper_length_mm,
        "sepal_width": body_mass_g,
        "petal_length": culmen_length_mm,
        "petal_width": culmen_depth_mm
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
            probs = data["probabilities"]  # [Adelie, Chinstrap, Gentoo]

            st.success(f"Predição: **{pred}**")
            st.write(f"Grau de certeza: **{confidence*100:.2f}%**")

            # Mostrar barras com as probabilidades
            st.subheader("Probabilidades por classe")
            labels = ["Adelie", "Chinstrap", "Gentoo"]
            # Implementa um DataFrame para exibir
            import pandas as pd
            df = pd.DataFrame({"classe": labels, "probabilidade": probs})
            df = df.set_index("classe")
            st.bar_chart(df)

            # Opcional: mostrar probabilidades de todas as classes
            st.write("Probabilidades (raw):", {labels[i]: f"{probs[i]*100:.2f}%" for i in range(len(labels))})
    except requests.exceptions.RequestException as e:
        st.error(f"Falha ao conectar com a API: {e}")
