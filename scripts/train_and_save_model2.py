
# scripts/train_and_save_model.py
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

def train_and_save(path="app_backend/model/penguim_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = sns.load_dataset("penguins").dropna()
    X = data[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]].values
    y = data["species"].astype("category").cat.codes.values
     # Modelo utilizado: Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
     # Salva no formato pickle
    with open(path, "wb") as f:
        pickle.dump(model, f)

    print(f"Modelo treinado e salvo em: {path}")

if __name__ == "__main__":
    train_and_save()
