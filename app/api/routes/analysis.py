from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
import os
import uuid
import shutil
import sys
from pathlib import Path

from app.core.config import settings
from app.core.database import get_db
from app.models.user import User
from app.models.patient import Patient, PatientImage
from app.api.routes.auth import get_current_user

router = APIRouter()

# Determine uploads directory correctly for both development and packaged environments
if getattr(sys, 'frozen', False):
    # We are running in a bundled app
    base_dir = Path(sys.executable).parent
else:
    # We are running in a normal Python environment
    base_dir = Path(__file__).resolve().parent.parent.parent.parent

UPLOAD_DIR = os.path.join(base_dir, 'uploads')
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    print(f"Created uploads directory in analysis route: {UPLOAD_DIR}")

@router.post("/{patient_uid}/upload")
async def upload_analysis_image(
    patient_uid: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if patient exists
    patient = db.query(Patient).filter(Patient.uid == patient_uid).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    # Check file type
    allowed_extensions = settings.ALLOWED_EXTENSIONS
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Generate unique filename
    unique_filename = f"{patient_uid}_{datetime.utcnow().timestamp()}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Save the file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving file: {str(e)}"
        )
    
    # Create URL that can be accessed via the /files endpoint
    file_url = f"/files/{unique_filename}"
    
    # Create image record
    image = PatientImage(
        patient_id=patient.id,
        url=file_url,  # Store the URL path instead of the physical file path
        uploaded_at=datetime.utcnow(),
        analysis_status="pending"
    )
    
    try:
        db.add(image)
        db.commit()
        db.refresh(image)
        
        return {
            "image_id": image.id,
            "status": "uploaded",
            "file_url": file_url,
            "message": "Image uploaded successfully. Analysis pending."
        }
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )

@router.get("/{analysis_id}")
async def get_analysis_result(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if analysis exists
    image = db.query(PatientImage).filter(PatientImage.id == analysis_id).first()
    if not image:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Check if patient exists and user has permission
    patient = db.query(Patient).filter(Patient.id == image.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if current_user.role != "admin" and patient.created_by != current_user.email:
        raise HTTPException(status_code=403, detail="Not authorized to view this analysis")
    
    # Return analysis result
    return {
        "id": image.id,
        "patient_uid": patient.uid,
        "status": image.analysis_status,
        "result": image.analysis_result or {},
        "upload_date": image.uploaded_at,
        "completion_date": image.analysis_completed_at
    } 