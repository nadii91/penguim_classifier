# app_backend/model_util.py
import os
import pickle
import numpy as np

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado em: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

# Assumimos que modelo tem predict() e predict_proba()
def predict_instance(model, x):
    """
    x: lista ou array com 4 floats
    retorna: (predicted_class_name, confidence, probabilities_array)
    """
    arr = np.array(x, dtype=float).reshape(1, -1)
    probs = model.predict_proba(arr)[0]  # probabilities for each class in model.classes_
    idx = int(probs.argmax())
    confidence = float(probs[idx])
    # Map numeric class to human name (sklearn iris uses 0=setosa,1=versicolor,2=virginica)
    class_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
    # If model.classes_ is not [0,1,2], we map according to model.classes_
    try:
        label = model.classes_[idx]
        # if classes are encoded as ints 0/1/2
        label_int = int(label)
        pred_name = class_map.get(label_int, str(label))
    except Exception:
        pred_name = str(model.predict(arr)[0])
    return pred_name, confidence, probs
