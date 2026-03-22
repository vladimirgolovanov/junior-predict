#import debugpy

#debugpy.listen(("0.0.0.0", 5678))
#print("Waiting for debugger attach on port 5678...")
#debugpy.wait_for_client()   # сервис подождёт пока PyCharm подключится

from fastapi import FastAPI
from pydantic import BaseModel
from .model import predict_end

app = FastAPI()


class PredictRequest(BaseModel):
    child_id: int
    start: str


@app.post("/predict")
def predict(req: PredictRequest):
    end = await predict_end(req.child_id, req.start)
    return {"predicted_end": end}
