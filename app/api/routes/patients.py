from fastapi import APIRouter, HTTPException, Depends, status
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import uuid
from ..dependencies import get_db
from app.models.user import UserInDB
from app.api.routes.auth import get_current_user

router = APIRouter()

class PatientBase(BaseModel):
    firstName: str = Field(..., description="Patient's first name")
    lastName: str = Field(..., description="Patient's last name")
    dateOfBirth: str = Field(..., description="Patient's date of birth")
    gender: str = Field(..., description="Patient's gender")
    contactNumber: str = Field(..., description="Patient's contact number")
    address: Optional[str] = Field(None, description="Patient's address")
    medicalHistory: Optional[str] = Field(None, description="Patient's medical history")

class PatientCreate(PatientBase):
    pass

class PatientResponse(PatientBase):
    uid: str = Field(..., description="Patient's unique identifier")
    created_at: datetime
    created_by: str

@router.post("/register", response_model=PatientResponse)
async def register_patient(
    patient: PatientCreate,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    try:
        # Generate unique patient ID
        uid = f"PT{uuid.uuid4().hex[:8].upper()}"
        
        # Create patient document
        patient_dict = patient.model_dump()
        patient_dict.update({
            "uid": uid,
            "created_at": datetime.utcnow(),
            "created_by": current_user.email
        })
        
        # Save to database
        result = await db["patients"].insert_one(patient_dict)
        
        if result.inserted_id:
            return PatientResponse(**patient_dict)
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to register patient"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/history", response_model=list[PatientResponse])
async def get_patients(
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    try:
        patients = await db["patients"].find().to_list(length=None)
        return [PatientResponse(**patient) for patient in patients]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 