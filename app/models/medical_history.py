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
    patient_uid: str = Field(..., description="Patient's unique identifier")
    file_type: str = Field(..., description="Type of file (e.g., xray, report)")
    file_name: str = Field(..., description="Original file name")
    file_path: str = Field(..., description="Path where file is stored")
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None
    uploaded_by: str = Field(..., description="Email of user who uploaded the file")
    doctor_name: str = Field(..., description="Name of the doctor")

    model_config = ConfigDict(
        populate_by_name=True,
        json_encoders={ObjectId: str},
        from_attributes=True
    )

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