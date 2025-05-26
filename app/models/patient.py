from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
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
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class ImageAnalysis(BaseModel):
    status: str = Field(..., description="Status of the analysis")
    result: dict = Field(default_factory=dict)
    completedAt: Optional[datetime] = None

class PatientImage(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    url: str
    uploadedAt: datetime = Field(default_factory=datetime.utcnow)
    analysis: Optional[ImageAnalysis] = None

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

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True

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