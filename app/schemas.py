from pydantic import BaseModel

class PredictionResponse(BaseModel):
    filename: str          # Matches your return key "filename"
    prediction: str        # Matches your return key "prediction"
    confidence: float
    db_id: int             # We add this so the ID is included in the response

    class Config:
        # This tells Pydantic to accept ORM objects (like your database row)
        # if you ever decide to return the database object directly.
        from_attributes = True