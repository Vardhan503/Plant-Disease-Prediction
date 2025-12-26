from .database import Base
from sqlalchemy import Column, String, Integer, Float
from sqlalchemy.sql.sqltypes import TIMESTAMP
from sqlalchemy.sql.expression import text

class Predictions(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, nullable=False)
    file_name = Column(String, nullable=False)
    class_name = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), nullable=False, server_default=text('now()'))