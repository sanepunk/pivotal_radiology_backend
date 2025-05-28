from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form, Response
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid
import os
from bson import ObjectId
from fastapi.responses import FileResponse
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
    name: str = Field(..., description="Patient's full name")
    date_of_birth: str = Field(..., description="Patient's date of birth")
    gender: str = Field(..., description="Patient's gender")
    contact: dict = Field(..., description="Patient's contact information")
    # email: str = Field(..., description="Patient's email")
    address: Optional[str] = Field(None, description="Patient's address")
    medical_history: Optional[str] = Field(None, description="Patient's medical history")

class PatientCreate(PatientBase):
    pass

class PatientResponse(PatientBase):
    uid: str = Field(..., description="Patient's unique identifier")
    created_at: datetime
    created_by: str

@router.post("/", response_model=PatientResponse)
async def create_patient(
    patient: PatientCreate,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    try:
        # Generate unique patient ID
        uid = f"PTB{uuid.uuid4().hex[:8].upper()}"
        
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
                detail="Failed to create patient"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=list[PatientResponse])
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

@router.get("/{uid}", response_model=PatientResponse)
async def get_patient(
    uid: str,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    patient = await db["patients"].find_one({"uid": uid})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PatientResponse(**patient)

@router.put("/{uid}", response_model=PatientResponse)
async def update_patient(
    uid: str,
    patient: PatientCreate,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    patient_dict = patient.model_dump()
    patient_dict["updated_at"] = datetime.utcnow()
    patient_dict["updated_by"] = current_user.email
    
    result = await db["patients"].update_one(
        {"uid": uid},
        {"$set": patient_dict}
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    updated_patient = await db["patients"].find_one({"uid": uid})
    return PatientResponse(**updated_patient)

@router.get("/{uid}/visits", response_model=List[VisitHistory])
async def get_patient_visits(
    uid: str,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    history = await db["visit_history"].find({"patient_uid": uid}).to_list(None)
    return history

@router.post("/{uid}/visits", response_model=VisitHistory)
async def add_visit(
    uid: str,
    visit: VisitHistory,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    # Verify patient exists
    patient = await db["patients"].find_one({"uid": uid})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    visit.patient_uid = uid
    visit.created_at = datetime.utcnow()
    visit.created_by = current_user.email
    
    result = await db["visit_history"].insert_one(visit.dict(exclude={"id"}))
    visit.id = str(result.inserted_id)
    return visit

@router.post("/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    patientUid: str = Form(...),
    notes: str = Form(None),
    doctor_name: str = Form(...),
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    # Verify patient exists
    patient = await db["patients"].find_one({"uid": patientUid})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Validate file type
    allowed_types = {
        'image/png': 'xray',
        'image/jpeg': 'xray',
        'application/dicom': 'xray',
        'application/pdf': 'report'
    }
    
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types are: {', '.join(allowed_types.keys())}"
        )

    # Create a unique filename
    filename = f"{patientUid}_{datetime.utcnow().timestamp()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    # Save the file
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Create file reference
    file_ref = FileReference(
        file_name=filename,
        file_path=filename,  # Store just the filename
        file_type=allowed_types[file.content_type],
        upload_date=datetime.utcnow(),
        uploaded_by=current_user.email,
        patient_uid=patientUid,
        doctor_name=doctor_name,
        notes=notes
    )
    
    # Save file reference to database
    result = await db["files"].insert_one(file_ref.dict(exclude={"id"}))
    file_ref.id = str(result.inserted_id)
    
    return {
        "id": file_ref.id,
        "filename": filename,
        "file_type": file_ref.file_type,
        "upload_date": file_ref.upload_date,
        "patient_uid": patientUid,
        "doctor_name": doctor_name
    }

@router.get("/files/{file_id}")
async def get_file(
    file_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    file_ref = await db["files"].find_one({"_id": ObjectId(file_id)})
    if not file_ref:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = file_ref["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    return FileResponse(
        file_path,
        media_type="image/jpeg" if file_path.endswith(('.jpg', '.jpeg')) else "image/png",
        filename=file_ref["file_name"]
    )

@router.get("/{uid}/files")
async def get_patient_files(
    uid: str,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    files = await db["files"].find({"patient_uid": uid}).to_list(None)
    # Convert ObjectId to string and format the response
    formatted_files = []
    for file in files:
        file['_id'] = str(file['_id'])
        formatted_files.append(file)
    return formatted_files 