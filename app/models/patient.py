from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional, List, Any
from datetime import datetime
from bson import ObjectId

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

class ImageAnalysis(BaseModel):
    status: str = Field(..., description="Status of the analysis")
    result: dict = Field(default_factory=dict)
    completedAt: Optional[datetime] = None

class PatientImage(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    url: str
    uploadedAt: datetime = Field(default_factory=datetime.utcnow)
    analysis: Optional[ImageAnalysis] = None

    model_config = ConfigDict(populate_by_name=True, json_encoders={ObjectId: str})

class PatientBase(BaseModel):
    name: str
    age: int
    sex: str
    mobile: str
    email: Optional[EmailStr] = None
    whatsapp: Optional[str] = None
    address: str
    location: str

class PatientCreate(PatientBase):
    pass

class PatientInDB(PatientBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)
    images: List[PatientImage] = Field(default_factory=list)

    model_config = ConfigDict(populate_by_name=True, json_encoders={ObjectId: str})

class PatientUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    mobile: Optional[str] = None
    email: Optional[EmailStr] = None
    whatsapp: Optional[str] = None
    address: Optional[str] = None
    location: Optional[str] = None

class PatientResponse(PatientInDB):
    pass

class Insurance(BaseModel):
    provider: Optional[str] = None
    policy_number: Optional[str] = None
    expiry_date: Optional[datetime] = None

class Contact(BaseModel):
    phone: str
    email: Optional[str]
    address: str
    emergency_contact: Optional[str]

class MedicalCondition(BaseModel):
    condition: str
    diagnosed_date: datetime
    status: str = "Active"
    notes: Optional[str]

class Medication(BaseModel):
    name: Optional[str] = None
    dosage: Optional[str] = None
    frequency: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    prescribing_doctor: Optional[str] = None

class Patient(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    uid: str = Field(...)
    name: str = Field(...)
    date_of_birth: datetime
    gender: str
    contact: Contact
    insurance: Insurance
    medical_conditions: List[MedicalCondition] = []
    medications: List[Medication] = []
    allergies: List[str] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(populate_by_name=True, json_encoders={ObjectId: str}) 