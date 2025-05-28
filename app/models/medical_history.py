from datetime import datetime
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
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

class Diagnosis(BaseModel):
    condition: str
    diagnosed_by: str
    notes: Optional[str]

class Procedure(BaseModel):
    name: str
    performed_by: str
    notes: Optional[str]
    date: datetime

class FileReference(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    patient_uid: str
    file_type: str  # e.g., "xray", "lab_report", "prescription"
    file_name: str
    file_path: str
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str]
    uploaded_by: str
    doctor_name: str

    model_config = ConfigDict(populate_by_name=True, json_encoders={ObjectId: str})

class VisitHistory(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    patient_uid: str
    visit_date: datetime
    doctor: str
    chief_complaint: str
    notes: str
    diagnoses: List[Diagnosis] = []
    procedures: List[Procedure] = []
    prescribed_medications: List[str] = []
    files: List[FileReference] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(populate_by_name=True, json_encoders={ObjectId: str}) 