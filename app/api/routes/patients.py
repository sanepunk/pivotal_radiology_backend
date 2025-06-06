from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form, Response
from sqlalchemy.orm import Session
from sqlalchemy.future import select
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import uuid
import os
import sys
from pathlib import Path
from fastapi.responses import FileResponse

from ..dependencies import get_db
from app.models.user import User
from app.api.routes.auth import get_current_user
from ...models.patient import Patient, PatientCreate, PatientInDB, PatientUpdate, PatientImageSchema, Contact, Insurance
from ...models.medical_history import VisitHistory, FileReference, VisitHistoryCreate, VisitHistoryInDB, FileReferenceSchema
from ...core.database import get_db
from app.core.config import settings

router = APIRouter()

# Determine uploads directory correctly for both development and packaged environments
if getattr(sys, 'frozen', False):
    # We are running in a bundled app - use consistent location in user's home
    base_dir = Path(os.path.expanduser("~")) / ".pivotal"
    os.makedirs(base_dir, exist_ok=True)
else:
    # We are running in a normal Python environment
    base_dir = Path(__file__).resolve().parent.parent.parent.parent

# For packaged app, use the persistent uploads directory
if getattr(sys, 'frozen', False):
    HOME_DIR = os.path.expanduser("~")
    PERSISTENT_DIR = os.path.join(HOME_DIR, '.pivotal')
    UPLOAD_DIR = os.path.join(PERSISTENT_DIR, 'uploads')
else:
    UPLOAD_DIR = os.path.join(base_dir, 'uploads')

# Create directory if it doesn't exist
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    print(f"Created uploads directory in patients route: {UPLOAD_DIR}")
else:
    print(f"Using existing uploads directory: {UPLOAD_DIR}")

# Compatibility model for old frontend format
class OldPatientCreate(BaseModel):
    name: str
    age: Optional[int] = None
    sex: Optional[str] = None
    mobile: Optional[str] = None
    email: Optional[str] = None
    whatsapp: Optional[str] = None
    address: Optional[str] = None
    location: Optional[str] = None
    date_of_birth: Optional[datetime] = None
    gender: Optional[str] = None
    contact: Optional[dict] = None
    medical_history: Optional[str] = None

# Helper function to convert SQLAlchemy model to Pydantic model
def convert_to_pydantic(db_patient: Patient) -> PatientInDB:
    # Extract contact from JSON
    contact_data = db_patient.contact_info or {}
    contact = Contact(
        phone=contact_data.get("phone", ""),
        email=contact_data.get("email"),
        address=contact_data.get("address", ""),
        emergency_contact=contact_data.get("emergency_contact")
    )
    
    # Extract insurance from JSON
    insurance_data = db_patient.insurance_info or {}
    insurance = Insurance(
        provider=insurance_data.get("provider"),
        policy_number=insurance_data.get("policy_number"),
        expiry_date=insurance_data.get("expiry_date")
    )
    
    # Create Pydantic model
    return PatientInDB(
        id=db_patient.id,
        uid=db_patient.uid,
        name=db_patient.name,
        date_of_birth=db_patient.date_of_birth,
        gender=db_patient.gender,
        contact=contact,
        insurance=insurance,
        medical_conditions=db_patient.medical_conditions or [],
        medications=db_patient.medications or [],
        allergies=db_patient.allergies or [],
        created_at=db_patient.created_at,
        updated_at=db_patient.updated_at,
        created_by=db_patient.created_by,
        images=[]
    )

@router.post("/", response_model=PatientInDB)
async def create_patient(
    patient_data: OldPatientCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Generate unique patient ID
        uid = f"PTB{uuid.uuid4().hex[:8].upper()}"
        
        # Handle compatibility with old frontend format
        if hasattr(patient_data, 'contact') and patient_data.contact:
            # New format with contact object
            contact_info = patient_data.contact
            gender = patient_data.gender
            dob = patient_data.date_of_birth
        else:
            # Old format with separate fields
            contact_info = {
                "phone": patient_data.mobile or "",
                "email": patient_data.email,
                "address": patient_data.address or "",
                "emergency_contact": patient_data.whatsapp
            }
            gender = patient_data.sex or "unknown"
            dob = patient_data.date_of_birth or datetime.now()
        
        # Create patient
        db_patient = Patient(
            uid=uid,
            name=patient_data.name,
            date_of_birth=dob,
            gender=gender,
            contact_info=contact_info,
            insurance_info={},
            medical_conditions=[],
            medications=[],
            allergies=[],
            created_at=datetime.utcnow(),
            created_by=current_user.email
        )
        
        db.add(db_patient)
        db.commit()
        db.refresh(db_patient)
        
        # Convert SQLAlchemy model to Pydantic model before returning
        return convert_to_pydantic(db_patient)
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=List[PatientInDB])
async def get_all_patients(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        stmt = select(Patient)
        
        # If not admin, only return patients created by this user
        if current_user.role != "admin":
            stmt = stmt.where(Patient.created_by == current_user.email)
            
        patients = db.execute(stmt).scalars().all()
        
        # Convert all SQLAlchemy models to Pydantic models
        return [convert_to_pydantic(patient) for patient in patients]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{patient_id}", response_model=PatientInDB)
async def get_patient(
    patient_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Check if the patient_id is a UUID or a UID
        if len(patient_id) == 11 and patient_id.startswith("PTB"):
            # It's a UID
            stmt = select(Patient).where(Patient.uid == patient_id)
        else:
            # Assume it's an ID
            stmt = select(Patient).where(Patient.id == patient_id)
        
        db_patient = db.execute(stmt).scalar_one_or_none()
        
        if not db_patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )
        
        return convert_to_pydantic(db_patient)
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/{patient_id}", response_model=PatientInDB)
async def update_patient(
    patient_id: str,
    patient_data: PatientUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Check if the patient_id is a UUID or a UID
        if len(patient_id) == 11 and patient_id.startswith("PTB"):
            # It's a UID
            stmt = select(Patient).where(Patient.uid == patient_id)
        else:
            # Assume it's an ID
            stmt = select(Patient).where(Patient.id == patient_id)
        
        db_patient = db.execute(stmt).scalar_one_or_none()
        
        if not db_patient:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )
        
        # Update patient fields
        if patient_data.name is not None:
            db_patient.name = patient_data.name
        if patient_data.date_of_birth is not None:
            db_patient.date_of_birth = patient_data.date_of_birth
        if patient_data.gender is not None:
            db_patient.gender = patient_data.gender
        if patient_data.contact is not None:
            db_patient.contact_info = {
                "phone": patient_data.contact.phone,
                "email": patient_data.contact.email,
                "address": patient_data.contact.address,
                "emergency_contact": patient_data.contact.emergency_contact
            }
        if patient_data.insurance is not None:
            db_patient.insurance_info = {
                "provider": patient_data.insurance.provider,
                "policy_number": patient_data.insurance.policy_number,
                "expiry_date": patient_data.insurance.expiry_date
            }
        if patient_data.medical_conditions is not None:
            db_patient.medical_conditions = patient_data.medical_conditions
        if patient_data.medications is not None:
            db_patient.medications = patient_data.medications
        if patient_data.allergies is not None:
            db_patient.allergies = patient_data.allergies
        
        db_patient.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_patient)
        
        return convert_to_pydantic(db_patient)
    
    except Exception as e:
        db.rollback()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{uid}/visits", response_model=List[VisitHistoryInDB])
async def get_patient_visits(
    uid: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    visits = db.query(VisitHistory).filter(VisitHistory.patient_uid == uid).all()
    return visits

@router.post("/{uid}/visits", response_model=VisitHistoryInDB)
async def add_visit(
    uid: str,
    visit: VisitHistoryCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify patient exists
    patient = db.query(Patient).filter(Patient.uid == uid).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    # Create new visit
    db_visit = VisitHistory(
        patient_uid=uid,
        visit_date=visit.visit_date,
        doctor=visit.doctor,
        chief_complaint=visit.chief_complaint,
        notes=visit.notes,
        diagnoses=[diagnosis.model_dump() for diagnosis in visit.diagnoses],
        procedures=[procedure.model_dump() for procedure in visit.procedures],
        prescribed_medications=visit.prescribed_medications,
        created_at=datetime.utcnow(),
        created_by=current_user.email
    )
    
    db.add(db_visit)
    db.commit()
    db.refresh(db_visit)
    return db_visit

@router.post("/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    patientUid: str = Form(...),
    doctor_name: str = Form(...),
    notes: Optional[str] = Form(default=None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Verify patient exists
        patient = db.query(Patient).filter(Patient.uid == patientUid).first()
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
        
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Allowed types are: {', '.join(allowed_types.keys())}"
            )

        # Create a unique filename
        timestamp = datetime.utcnow().timestamp()
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
        filename = f"{patientUid}_{timestamp}_{safe_filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Create URL for accessing via the /files endpoint
        file_url = f"/files/{filename}"
        
        try:
            # Save the file
            content = await file.read()
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            # Create file reference
            db_file = FileReference(
                patient_uid=patientUid,
                file_type=allowed_types[file.content_type],
                file_name=filename,
                file_path=file_url,  # Store URL path instead of filesystem path
                upload_date=datetime.utcnow(),
                uploaded_by=current_user.email,
                doctor_name=doctor_name,
                notes=notes
            )
            
            db.add(db_file)
            db.commit()
            db.refresh(db_file)
            
            return {
                "id": db_file.id,
                "filename": filename,
                "file_url": file_url,
                "file_type": db_file.file_type,
                "upload_date": db_file.upload_date,
                "patient_uid": patientUid,
                "doctor_name": doctor_name
            }
            
        except Exception as e:
            # Clean up the file if something goes wrong
            if os.path.exists(file_path):
                os.remove(file_path)
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error saving file: {str(e)}"
            )
            
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Error processing request: {str(e)}"
        )

@router.get("/files/{file_id}", response_model=FileReferenceSchema)
async def get_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file_ref = db.query(FileReference).filter(FileReference.id == file_id).first()
    if not file_ref:
        raise HTTPException(status_code=404, detail="File not found")
    
    return file_ref

@router.get("/files/{file_id}/download")
async def download_file(
    file_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    file_ref = db.query(FileReference).filter(FileReference.id == file_id).first()
    if not file_ref:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Extract the filename from the stored URL path
    filename = os.path.basename(file_ref.file_path)
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Determine content type based on file type
    content_type = None
    if file_ref.file_type == "xray":
        if file_path.lower().endswith(('.jpg', '.jpeg')):
            content_type = "image/jpeg"
        elif file_path.lower().endswith('.png'):
            content_type = "image/png"
        elif file_path.lower().endswith('.dcm'):
            content_type = "application/dicom"
    elif file_ref.file_type == "report":
        content_type = "application/pdf"
    
    if not content_type:
        content_type = "application/octet-stream"
    
    return FileResponse(
        file_path,
        media_type=content_type,
        filename=filename
    )

@router.get("/{uid}/files", response_model=List[FileReferenceSchema])
async def get_patient_files(
    uid: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # First check if user has access to this patient
    patient = db.query(Patient).filter(Patient.uid == uid).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    if current_user.role != "admin" and patient.created_by != current_user.email:
        raise HTTPException(status_code=403, detail="Not authorized to view this patient's files")
    
    files = db.query(FileReference).filter(FileReference.patient_uid == uid).all()
    return files 