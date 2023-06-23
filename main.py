from fastapi import FastAPI, Request
from model.config import Config

from model.schema.symptom import SymptomScheme
from utils import predict

app = FastAPI()
config = Config.load()


@app.post("/symptoscan/")
def say_hello(request: Request, symptom: SymptomScheme):
    prediction = predict(config, symptom.dict())
    print(prediction)
    return {
        "disease":prediction[0],
        "confidence":prediction[1]
    }

