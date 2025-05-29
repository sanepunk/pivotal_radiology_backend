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
        # If user is admin, return all patients
        if current_user.role == "admin":
            patients = await db["patients"].find().to_list(length=None)
        else:
            # If user is doctor, return only their patients
            patients = await db["patients"].find({"created_by": current_user.email}).to_list(length=None)
        
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
    
    # Check if user has access to this patient
    if current_user.role != "admin" and patient["created_by"] != current_user.email:
        raise HTTPException(status_code=403, detail="Not authorized to view this patient")
    
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
    patientUid: str = Form(...),
    doctor_name: str = Form(...),
    file_path: str = Form(...),
    file_name: str = Form(...),
    file_type: str = Form(...),
    notes: Optional[str] = Form(default=None),
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    try:
        # Debug logging
        print(f"Received upload request - patientUid: {patientUid}, doctor_name: {doctor_name}")
        print(f"File info - filename: {file_name}, file_type: {file_type}")
        
        # Verify patient exists
        patient = await db["patients"].find_one({"uid": patientUid})
        if not patient:
            raise HTTPException(status_code=404, detail=f"Patient with UID {patientUid} not found")

        # Validate file type
        allowed_types = {
            'image/png': 'xray',
            'image/jpeg': 'xray',
            'image/jpg': 'xray',
            'application/dicom': 'xray',
            'application/pdf': 'report'
        }
        
        if file_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file_type}. Allowed types are: {', '.join(allowed_types.keys())}"
            )

        # Create file reference
        file_data = {
            "patient_uid": patientUid,
            "file_type": allowed_types[file_type],
            "file_name": file_name,
            "file_path": file_path,
            "upload_date": datetime.utcnow(),
            "uploaded_by": current_user.email,
            "doctor_name": doctor_name,
            "notes": notes
        }
        
        print(f"Creating FileReference with data: {file_data}")
        
        # Create and validate FileReference object
        file_ref = FileReference(**file_data)
        
        # Save to database
        result = await db["files"].insert_one(file_ref.model_dump(exclude={"id"}))
        file_ref.id = str(result.inserted_id)
        
        return {
            "id": str(result.inserted_id),
            "filename": file_name,
            "file_type": file_ref.file_type,
            "upload_date": file_ref.upload_date,
            "patient_uid": patientUid,
            "doctor_name": doctor_name,
            "file_path": file_path
        }
            
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=422,
            detail=f"Error processing request: {str(e)}"
        )

@router.get("/files/{file_id}")
async def get_file(
    file_id: str,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    file_ref = await db["files"].find_one({"_id": ObjectId(file_id)})
    if not file_ref:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Return a redirect to the Supabase URL
    return Response(
        status_code=307,  # Temporary redirect
        headers={"Location": file_ref["file_path"]}
    )

@router.get("/{uid}/files")
async def get_patient_files(
    uid: str,
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    # First check if user has access to this patient
    patient = await db["patients"].find_one({"uid": uid})
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if current_user.role != "admin" and patient["created_by"] != current_user.email:
        raise HTTPException(status_code=403, detail="Not authorized to view this patient's files")
    
    files = await db["files"].find({"patient_uid": uid}).to_list(None)
    formatted_files = []
    for file in files:
        file['_id'] = str(file['_id'])
        formatted_files.append(file)
    return formatted_files 