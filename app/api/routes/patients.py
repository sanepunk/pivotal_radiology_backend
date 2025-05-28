from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid
import os
from bson import ObjectId
from ..dependencies import get_db
from app.models.user import UserInDB
from app.api.routes.auth import get_current_user
from ...models.patient import Patient
from ...models.medical_history import VisitHistory, FileReference
from ...core.database import get_database

router = APIRouter()
db = get_database()

UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

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
            "created_at": datetime.now(),
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

@router.get("/patients/{uid}", response_model=Patient)
async def get_patient(uid: str):
    patient = await db.patients.find_one({"uid": uid})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@router.put("/patients/{uid}", response_model=Patient)
async def update_patient(uid: str, patient: Patient):
    patient.updated_at = datetime.utcnow()
    result = await db.patients.update_one(
        {"uid": uid},
        {"$set": patient.dict(exclude={"id"})}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient

@router.get("/patients/{uid}/history", response_model=List[VisitHistory])
async def get_patient_history(uid: str):
    history = await db.visit_history.find({"patient_uid": uid}).to_list(None)
    return history

@router.post("/patients/{uid}/history", response_model=VisitHistory)
async def add_visit_history(uid: str, visit: VisitHistory):
    visit.patient_uid = uid
    visit.created_at = datetime.utcnow()
    visit.updated_at = datetime.utcnow()
    result = await db.visit_history.insert_one(visit.dict(exclude={"id"}))
    visit.id = result.inserted_id
    return visit

@router.post("/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    patientUid: str = Form(...),
    type: str = Form(...)
):
    # Create a unique filename
    filename = f"{patientUid}_{datetime.utcnow().timestamp()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Create file reference
    file_ref = FileReference(
        file_type=type,
        file_name=filename,
        file_path=file_path,
        upload_date=datetime.utcnow()
    )
    
    # Save file reference to database
    result = await db.files.insert_one(file_ref.dict(exclude={"id"}))
    file_ref.id = result.inserted_id
    
    # Add file reference to latest visit history
    latest_visit = await db.visit_history.find_one(
        {"patient_uid": patientUid},
        sort=[("created_at", -1)]
    )
    
    if latest_visit:
        await db.visit_history.update_one(
            {"_id": latest_visit["_id"]},
            {"$push": {"files": file_ref.dict()}}
        )
    
    return {"id": str(file_ref.id), "filename": filename}

@router.get("/files/{file_id}")
async def get_file(file_id: str):
    file_ref = await db.files.find_one({"_id": ObjectId(file_id)})
    if not file_ref:
        raise HTTPException(status_code=404, detail="File not found")
    
    if not os.path.exists(file_ref["file_path"]):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    return FileResponse(file_ref["file_path"]) 