# app_backend/api.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from fastapi.middleware.cors import CORSMiddleware
from .model_util import load_model, predict_instance

app = FastAPI(title="Iris Predictor API")

# Permissões do Streamlit (rodando em outra porta) para chamar API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Não Restrinje os domínios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input schema
class IrisInput(BaseModel):
    sepal_length: float = Field(..., gt=0)
    sepal_width: float = Field(..., gt=0)
    petal_length: float = Field(..., gt=0)
    petal_width: float = Field(..., gt=0)

# Output schema
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float  # 0..1
    probabilities: List[float]  # probabilidades para [setosa, versicolor, virginica]


# Load model on startup
MODEL_PATH = "app_backend/model/iris_model.pkl"
model = load_model(MODEL_PATH)


@app.get("/")
def read_root():
    return {"message": "Iris Predictor API. POST to /predict with sepal/petal measurements."}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: IrisInput):
    x = [
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width,
    ]
    pred_class, confidence, probs = predict_instance(model, x)
    return PredictionResponse(
        predicted_class=pred_class,
        confidence=round(confidence, 4),
        probabilities=[round(float(p), 4) for p in probs.tolist()]
    )
