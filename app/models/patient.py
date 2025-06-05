from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any

from app.core.database import Base

# SQLAlchemy Models
class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    uid = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    date_of_birth = Column(DateTime, nullable=False)
    gender = Column(String, nullable=False)
    contact_info = Column(JSON, nullable=False)  # Store Contact as JSON
    insurance_info = Column(JSON)  # Store Insurance as JSON
    medical_conditions = Column(JSON, default=list)  # Store list of conditions as JSON
    medications = Column(JSON, default=list)  # Store medications as JSON
    allergies = Column(JSON, default=list)  # Store allergies as JSON list
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by = Column(String, ForeignKey("users.email"))

    # Define relationships
    images = relationship("PatientImage", back_populates="patient", cascade="all, delete-orphan")
    creator = relationship("User", foreign_keys=[created_by])
    
class PatientImage(Base):
    __tablename__ = "patient_images"
    
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    url = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    analysis_status = Column(String)
    analysis_result = Column(JSON)  # Store analysis result as JSON
    analysis_completed_at = Column(DateTime, nullable=True)
    
    # Define relationships
    patient = relationship("Patient", back_populates="images")

# Pydantic Models for validation and responses
class Contact(BaseModel):
    phone: str
    email: Optional[str] = None
    address: str
    emergency_contact: Optional[str] = None
    
    class Config:
        from_attributes = True

class Insurance(BaseModel):
    provider: Optional[str] = None
    policy_number: Optional[str] = None
    expiry_date: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class MedicalCondition(BaseModel):
    condition: str
    diagnosed_date: datetime
    status: str = "Active"
    notes: Optional[str] = None
    
    class Config:
        from_attributes = True

class Medication(BaseModel):
    name: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    prescribing_doctor: Optional[str] = None
    
    class Config:
        from_attributes = True

class ImageAnalysis(BaseModel):
    status: str
    result: Dict[str, Any] = {}
    completedAt: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class PatientImageSchema(BaseModel):
    id: int
    url: str
    uploaded_at: datetime
    analysis_status: Optional[str] = None
    analysis_result: Optional[Dict[str, Any]] = None
    analysis_completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Base schema for patient operations
class PatientBase(BaseModel):
    name: str
    date_of_birth: datetime
    gender: str
    contact: Contact
    insurance: Optional[Insurance] = None
    medical_conditions: List[MedicalCondition] = []
    medications: List[Medication] = []
    allergies: List[str] = []
    
    class Config:
        from_attributes = True

# Schema for creating patients
class PatientCreate(PatientBase):
    pass

# Schema for updating patients
class PatientUpdate(BaseModel):
    name: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    contact: Optional[Contact] = None
    insurance: Optional[Insurance] = None
    medical_conditions: Optional[List[MedicalCondition]] = None
    medications: Optional[List[Medication]] = None
    allergies: Optional[List[str]] = None
    
    class Config:
        from_attributes = True

# Schema for patient response
class PatientInDB(PatientBase):
    id: int
    uid: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    images: List[PatientImageSchema] = []
    
    class Config:
        from_attributes = True 