from pandas import DataFrame
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException´
import boto3
import os
import datetime
import json
 
model = load("pipeline.joblib")
 
app = FastAPI(version="1.0.0", title="Student Performance Prediction API", description="API to predict student performance based on various features."y)

class DataPredict(BaseModel):
    data_to_predict: list[dict]

@app.post("/predict")
def predict(request: DataPredict):
    try:
        list_data = request.data_to_predict
        df_data= DataFrame(list_data, columns=[
                                    'StudentID', 'Age', 'Gender', 'Ethnicity', 'ParentalEducation',
                                    'StudyTimeWeekly', 'Absences', 'Tutoring', 'ParentalSupport',
                                    'Extracurricular', 'Sports', 'Music', 'Volunteering', 'GPA'
                                    ])
 
        prediction = model.predict(df_data)
        output = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "input_data": request.data_to_predict,
            "prediction": prediction.tolist()
        }
        saved_path = save_prediction_s3(output)
        output["s3_path"] = saved_path
        return output
        
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/sum")
def sum(param1: float, param2: float):
    try:
        result = param1 + param2
        return {"param1": param1, "param2": param2, "sum": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {'Universidad EIA': 'MLOps-Taller 3'}
 