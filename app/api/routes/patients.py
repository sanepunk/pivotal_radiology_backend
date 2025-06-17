from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form, Response
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid
import os
from bson import ObjectId
import sys
from fastapi.responses import FileResponse
from ..dependencies import get_db
from app.models.user import UserInDB
from app.api.routes.auth import get_current_user
from ...models.patient import Patient
from ...models.medical_history import VisitHistory, FileReference
from ...core.database import get_database
from ...core.dicom_utils import convert_dicom_to_png

router = APIRouter()
db = get_database()

# Define directories for different file types
UPLOAD_DIR = "uploads"
DICOM_DIR = os.path.join(UPLOAD_DIR, "dicom")
IMAGE_DIR = os.path.join(UPLOAD_DIR, "image")

# Create directories if they don't exist
for directory in [UPLOAD_DIR, DICOM_DIR, IMAGE_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

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
    file: UploadFile = File(...),
    patientUid: str = Form(...),
    doctor_name: str = Form(...),
    notes: Optional[str] = Form(default=None),
    is_dicom: bool = Form(default=False),
    current_user: UserInDB = Depends(get_current_user),
    db = Depends(get_db)
):
    try:
        # Debug logging
        print(f"Received upload request - patientUid: {patientUid}, doctor_name: {doctor_name}")
        print(f"File info - filename: {file.filename}, content_type: {file.content_type}, is_dicom: {is_dicom}")
        
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
            'image/x-dicom': 'xray',  # Additional MIME type for DICOM
            'application/pdf': 'report'
        }
        
        if file.content_type not in allowed_types and not is_dicom:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Allowed types are: {', '.join(allowed_types.keys())}"
            )

        # Create a unique timestamp for filenames
        timestamp = datetime.utcnow().timestamp()
        
        # Read file content
        content = await file.read()
        
        # Handle DICOM files specially
        if is_dicom or file.content_type in ['application/dicom', 'image/x-dicom']:
            try:
                # Convert DICOM to PNG
                png_data, _, metadata = convert_dicom_to_png(content, patientUid)
                
                # Create consistent filenames with uid and timestamp
                dicom_filename = f"{patientUid}_{timestamp}_original.dcm"
                png_filename = f"{patientUid}_{timestamp}.png"
                
                # Save original DICOM file
                dicom_path = os.path.join(DICOM_DIR, dicom_filename)
                with open(dicom_path, "wb") as buffer:
                    buffer.write(content)
                
                # Save converted PNG
                png_path = os.path.join(IMAGE_DIR, png_filename)
                with open(png_path, "wb") as buffer:
                    buffer.write(png_data)
                
                # Use the PNG as the main file reference
                filename = png_filename
                file_path = png_path
                
                # Additional metadata for notes
                patient_id = metadata.get('patient_id', 'Unknown')
                modality = metadata.get('modality', 'Unknown')
                additional_notes = f"{notes or ''} [DICOM: {dicom_filename}] [Patient ID: {patient_id}] [Modality: {modality}]"
                
                print(f"Saved DICOM to: {dicom_path}")
                print(f"Saved PNG to: {png_path}")
                
                # Create file reference with DICOM info
                file_data = {
                    "patient_uid": patientUid,
                    "file_type": 'xray',
                    "file_name": filename,
                    "file_path": file_path,
                    "upload_date": datetime.utcnow(),
                    "uploaded_by": current_user.email,
                    "doctor_name": doctor_name,
                    "notes": additional_notes
                }
                
            except Exception as e:
                print(f"Error processing DICOM file: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing DICOM file: {str(e)}"
                )
        else:
            # Regular file handling
            # Create a unique filename
            safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
            filename = f"{patientUid}_{timestamp}_{safe_filename}"
            
            # Choose appropriate directory based on file type
            if file.content_type.startswith('image/'):
                file_path = os.path.join(IMAGE_DIR, filename)
            else:
                file_path = os.path.join(UPLOAD_DIR, filename)
                
            # Save the file
            with open(file_path, "wb") as buffer:
                buffer.write(content)
                
            # Create file reference
            file_data = {
                "patient_uid": patientUid,
                "file_type": allowed_types.get(file.content_type, 'document'),
                "file_name": filename,
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
        
        # Generate URL paths for frontend
        base_url = "http://localhost:8000"  # Should come from configuration
        file_url = f"/files/{file_ref.id}"
        preview_url = f"{base_url}{file_url}"
        
        return {
            "id": str(result.inserted_id),
            "filename": filename,
            "file_path": file_path,
            "file_url": file_url,
            "preview_url": preview_url,
            "file_type": file_ref.file_type,
            "upload_date": file_ref.upload_date,
            "patient_uid": patientUid,
            "doctor_name": doctor_name
        }
            
    except Exception as e:
        # Clean up any created files if something goes wrong
        print(f"Error during file processing: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error saving file: {str(e)}"
        )
            
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
    
    file_path = file_ref["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Determine content type based on file type
    content_type = None
    if file_ref["file_type"] == "xray":
        if file_path.lower().endswith(('.jpg', '.jpeg')):
            content_type = "image/jpeg"
        elif file_path.lower().endswith('.png'):
            content_type = "image/png"
        elif file_path.lower().endswith('.dcm'):
            content_type = "application/dicom"
    elif file_ref["file_type"] == "report":
        content_type = "application/pdf"
    
    if not content_type:
        content_type = "application/octet-stream"
    
    return FileResponse(
        file_path,
        media_type=content_type,
        filename=file_ref["file_name"]
    )

@router.get("/files/image/{filename}")
async def get_image_file(
    filename: str,
    current_user: UserInDB = Depends(get_current_user)
):
    """Serve image files directly from the image directory"""
    file_path = os.path.join(IMAGE_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image file not found")
    
    # Determine content type based on extension
    content_type = "application/octet-stream"
    if filename.lower().endswith(('.jpg', '.jpeg')):
        content_type = "image/jpeg"
    elif filename.lower().endswith('.png'):
        content_type = "image/png"
    
    return FileResponse(
        file_path,
        media_type=content_type,
        filename=filename
    )

@router.get("/files/dicom/{filename}")
async def get_dicom_file(
    filename: str,
    current_user: UserInDB = Depends(get_current_user)
):
    """Serve DICOM files directly from the DICOM directory"""
    file_path = os.path.join(DICOM_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="DICOM file not found")
    
    return FileResponse(
        file_path,
        media_type="application/dicom",
        filename=filename
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