from fastapi import FastAPI, Request
import uvicorn
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
    
    
if __name__=="__main__":
    uvicorn.run("main:app",host='0.0.0.0', port=4557, reload=True, workers=3)
    
    


