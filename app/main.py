from fastapi import FastAPI, Depends, UploadFile, status, HTTPException, File
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

from . import models, inference, schemas
from .database import get_db, engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    inference.load_model()
    # Create tables if they don't exist
    models.Base.metadata.create_all(bind=engine)
    yield
    # Shutdown
    print("Shutting down")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Plant Disease API is running"}

@app.post("/predict", status_code=status.HTTP_200_OK, response_model=schemas.PredictionResponse)
async def predict_plant_disease(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Upload an Image")
    
    contents = await file.read()
    
    try:
        class_name, confidence = inference.predict(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    db_prediction = models.Predictions(
        file_name=file.filename,  
        class_name=class_name,
        confidence=confidence
    )
    
    db.add(db_prediction)
    db.commit()
    db.refresh(db_prediction)
    
    return {
        "filename": file.filename,
        "prediction": class_name,
        "confidence": round(confidence, 4),
        "db_id": db_prediction.id
    }