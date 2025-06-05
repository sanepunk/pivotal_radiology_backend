from datetime import datetime
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from bson import ObjectId
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import relationship

from app.core.database import Base

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
        json_schema = handler(core_schema)
        json_schema.update(type="string")
        return json_schema

class Diagnosis(BaseModel):
    condition: str
    diagnosed_by: str
    notes: Optional[str] = None

    class Config:
        from_attributes = True

class Procedure(BaseModel):
    name: str
    performed_by: str
    notes: Optional[str] = None
    date: datetime

    class Config:
        from_attributes = True

class FileReference(Base):
    __tablename__ = "file_references"
    
    id = Column(Integer, primary_key=True, index=True)
    visit_id = Column(Integer, ForeignKey("visit_histories.id"), nullable=True)
    patient_uid = Column(String, index=True, nullable=False)
    file_type = Column(String, nullable=False)  # e.g., xray, report
    file_name = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    upload_date = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
    uploaded_by = Column(String, ForeignKey("users.email"), nullable=False)
    doctor_name = Column(String, nullable=False)
    
    # Define relationships
    visit = relationship("VisitHistory", back_populates="files")
    uploader = relationship("User", foreign_keys=[uploaded_by])

class VisitHistory(Base):
    __tablename__ = "visit_histories"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_uid = Column(String, index=True, nullable=False)
    visit_date = Column(DateTime, nullable=False)
    doctor = Column(String, nullable=False)
    chief_complaint = Column(String, nullable=False)
    notes = Column(Text, nullable=False)
    diagnoses = Column(JSON, default=list)  # Store list of diagnoses as JSON
    procedures = Column(JSON, default=list)  # Store list of procedures as JSON
    prescribed_medications = Column(JSON, default=list)  # Store list of medications as strings
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, ForeignKey("users.email"))
    
    # Define relationships
    files = relationship("FileReference", back_populates="visit", cascade="all, delete-orphan")
    creator = relationship("User", foreign_keys=[created_by])

class FileReferenceCreate(BaseModel):
    patient_uid: str
    file_type: str
    notes: Optional[str] = None
    doctor_name: str
    
    class Config:
        from_attributes = True

class FileReferenceSchema(BaseModel):
    id: int
    patient_uid: str
    file_type: str  
    file_name: str
    file_path: str
    upload_date: datetime
    notes: Optional[str] = None
    uploaded_by: str  
    doctor_name: str
    
    class Config:
        from_attributes = True

class VisitHistoryCreate(BaseModel):
    visit_date: datetime
    doctor: str
    chief_complaint: str
    notes: str
    diagnoses: List[Diagnosis] = []
    procedures: List[Procedure] = []
    prescribed_medications: List[str] = []
    
    class Config:
        from_attributes = True

class VisitHistoryInDB(VisitHistoryCreate):
    id: int
    patient_uid: str
    files: List[FileReferenceSchema] = []
    created_at: datetime
    updated_at: datetime
    created_by: str
    
    class Config:
        from_attributes = True 