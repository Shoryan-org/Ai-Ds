import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "blood_donor_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_availability(donor_data):
    model = load_model()

    data = pd.DataFrame([donor_data])

    prediction = model.predict(data)[0]

    return prediction