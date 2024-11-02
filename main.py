from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from keras.models import load_model
import autokeras as ak
import numpy as np
import joblib

loaded_model = load_model("./model_autokeras.keras", custom_objects=ak.CUSTOM_OBJECTS)
encoder = joblib.load('./label_encoder.pkl')
class_names = encoder.classes_

def get_pred(text):
    result = loaded_model.predict(np.array([text]))
    predicted_class = np.argmax(result, axis=-1)
    predicted_category = class_names[predicted_class[0]]
    return predicted_category

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    text: str

@app.get("/test")
def get_test():
    return {"Hello": "World"}

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        prediction = get_pred(request.text)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))