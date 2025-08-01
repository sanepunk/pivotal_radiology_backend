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
from ...models.patient import Patient, PatientCreate, PatientInDB, PatientUpdate, PatientImageSchema, Contact, Insurance, PatientImage
from ...models.medical_history import VisitHistory, FileReference, VisitHistoryCreate, VisitHistoryInDB, FileReferenceSchema
from ...core.database import get_db
from app.core.config import settings
from app.core.dicom_utils import convert_dicom_to_png
# from app.core.tb_model_enhanced import tb_model_enhanced as tb_model
# from app.core.tb_model import tb_model
from app.core.classifier import predict as classifier_predict
from app.core.three_d import predict as d_predict
# seg_model, cls_model = load()
router = APIRouter()
import pyvista as pv

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

# Create separate directories for DICOM and regular images
DICOM_DIR = os.path.join(UPLOAD_DIR, 'dicom')
IMAGE_DIR = os.path.join(UPLOAD_DIR, 'image')
# SEGMENTATION_DIR = os.path.join(UPLOAD_DIR, 'segmentation')

if not os.path.exists(DICOM_DIR):
    os.makedirs(DICOM_DIR)
    print(f"Created DICOM directory: {DICOM_DIR}")

if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)
    print(f"Created image directory: {IMAGE_DIR}")

# if not os.path.exists(SEGMENTATIO_DIR):
#     os.makedirs(SEGMENTATION_DIR)
#     print(f"Created segmentation directoNry: {SEGMENTATION_DIR}")

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
        stmt = select(Patient).order_by(Patient.created_at.desc())
        
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
    is_dicom: bool = Form(default=False),
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
            'image/x-dicom': 'xray',  # Additional MIME type for DICOM
            'application/pdf': 'report',
            'application/octet-stream': 'vti',  # For VTI 3D visualization files
            'application/vti': 'vti',  # For VTI 3D visualization files
            "image/vti": "vti"
        }
        
        if file.content_type not in allowed_types and not is_dicom:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Allowed types are: {', '.join(allowed_types.keys())}"
            )

        # Create a unique timestamp
        timestamp = datetime.utcnow().timestamp()
        
                # Read file content
        content = await file.read() 
        
        # Get patient data directly from Patient table
        patient_data = db.query(Patient).filter(Patient.uid == patientUid).first()
        gender = patient_data.gender.strip().lower()
        # Handle DICOM files specially
        if is_dicom or file.content_type in ['application/dicom', 'image/x-dicom']:
            try:
                print(1)
                # Convert DICOM to PNG
                png_data, _, metadata = convert_dicom_to_png(content, patientUid)
                print(2)
                # Create consistent filenames with uid and timestamp
                dicom_filename = f"{patientUid}_{timestamp}_original.dcm"
                png_filename = f"{patientUid}_{timestamp}.png"
                
                # Save original DICOM file
                dicom_path = os.path.join(DICOM_DIR, dicom_filename)
                with open(dicom_path, "wb") as buffer:
                    buffer.write(content)
                print(3)
                # Save converted PNG
                png_path = os.path.join(IMAGE_DIR, png_filename)
                with open(png_path, "wb") as buffer:
                    buffer.write(png_data)
                
                # Use the PNG for display purposes
                filename = png_filename
                
                # Create URL paths for references
                file_url = f"/files/image/{filename}"
                dicom_reference = f"dicom/{dicom_filename}"
                
                # Get metadata
                patient_id = metadata.get('patient_id', 'Unknown')
                modality = metadata.get('modality', 'Unknown')
                # segmentation_file = os.path.join(SEGMENTATION_DIR, f"{patientUid}_{timestamp}_segmentation.png")
                # Get TB prediction
                print(4)
                tb_prediction = classifier_predict(png_path)
                output_files = tb_prediction['output_files']
                cam_overlays = output_files['cam_overlays']
                segmentation_masks = output_files['segmentation_masks']
                heatmap_overlays = output_files['heatmap_overlays']
                print(tb_prediction['output_files'])

                print(tb_prediction)
                print(1000)
                tb_prediction = d_predict(png_path,tb_prediction['result']['Prediction_Class'],tb_prediction['result']['Confidence'], gender)[0]
                print(10000)
                output_files = tb_prediction.get('output_files', None)
                heatmap_severity_overlays = output_files.get('severity_url', None)
                rendering_3d = output_files.get('3d_rendering_url', None)
                rendering_png = output_files.get('rendering_png_url', None)
                print(tb_prediction['output_files'])
                
                # Create file reference with additional DICOM info and TB prediction
                db_file = FileReference(
                    patient_uid=patientUid,
                    file_type='xray',
                    file_name=filename,
                    file_path=file_url,
                    upload_date=datetime.utcnow(),
                    uploaded_by=current_user.email,
                    doctor_name=doctor_name,
                    notes=f"{notes or ''} [DICOM: {dicom_reference}] [Patient ID: {patient_id}] [Modality: {modality}] [TB Prediction: {"TB Positive" if tb_prediction['Prediction'] == "TB" else "TB Negative"} ({tb_prediction['Confidence']*100:.2f}%)]",
                    severity_overlay = heatmap_severity_overlays,
                    segmentation_mask = segmentation_masks,
                    heatmap_overlay = heatmap_overlays,
                    cam_overlay = cam_overlays,
                    rendering_3d = rendering_3d,
                    rendering_png = rendering_png
                )
                
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing DICOM file: {str(e)}"
                )
        else:
            # Regular file handling
            # Create a unique filename with uid and timestamp
            safe_filename = "".join(c for c in file.filename if c.isalnum() or c in "._-")
            filename = f"{patientUid}_{timestamp}_{safe_filename}"
            
            # Store in the image directory
            file_path = os.path.join(IMAGE_DIR, filename)
            
            # Create URL for accessing via the /files endpoint
            file_url = f"/files/image/{filename}"
            
            # Save the file
            with open(file_path, "wb") as buffer:
                buffer.write(content)
            
            # Get TB prediction for image files
            tb_prediction = None
            if file.content_type in ['image/png', 'image/jpeg', 'image/jpg']:
                try:
                    tb_prediction = classifier_predict(file_path)
                    print(tb_prediction)
                    output_files = tb_prediction.get('output_files', {})
                    cam_overlays = output_files.get('cam_overlays', None)
                    segmentation_masks = output_files.get('segmentation_masks', None)
                    heatmap_overlays = output_files.get('heatmap_overlays', None)
                    tb_prediction = d_predict(file_path,tb_prediction['result']['Prediction_Class'],tb_prediction['result']['Confidence'], gender)[0]
                    print(tb_prediction)
                    output_files = tb_prediction.get('output_files', {})
                    heatmap_severity_overlays = output_files.get('severity_url', None)
                    rendering_3d = output_files.get('3d_rendering_url', None)
                    rendering_png = output_files.get('rendering_png_url', None)
                except Exception as e:
                    print(f"Error getting TB prediction: {str(e)}")
            print(32)
            print(tb_prediction)
            print(f"{notes or ''} {f'[TB Prediction: {tb_prediction['Prediction']} ({tb_prediction['Confidence']*100:.2f}%)]' if tb_prediction else ''}")
            # Create file reference with TB prediction if available
            db_file = FileReference(
                patient_uid=patientUid,
                file_type=allowed_types.get(file.content_type, 'document'),
                file_name=filename,
                file_path=file_url,
                upload_date=datetime.utcnow(),
                uploaded_by=current_user.email,
                doctor_name=doctor_name,
                notes=f"{notes or ''} {f'[TB Prediction: {"TB Positive" if tb_prediction['Prediction'] == "TB" else "TB Negative"} ({tb_prediction['Confidence']*100:.2f}%)]' if tb_prediction else ''}",
                severity_overlay = heatmap_severity_overlays,
                segmentation_mask = segmentation_masks,
                heatmap_overlay = heatmap_overlays,
                cam_overlay = cam_overlays,
                rendering_3d = rendering_3d,
                rendering_png = rendering_png
            )
            print(33)
        print(33.5)
        try:
            print(34)
            db.add(db_file)
            db.commit()
            db.refresh(db_file)
            print(35)
            # Create a fully qualified URL for frontend preview
            if getattr(sys, 'frozen', False):
                # Production build - use the base URL from settings
                base_url = "http://localhost:8000"  # This should ideally come from settings
            else:
                # Development
                base_url = "http://localhost:8000"
                
            preview_url = f"{base_url}{file_url}"
            print(36)
            print({
                "id": db_file.id,
                "filename": filename,
                "file_url": file_url,
                "preview_url": preview_url,
                "file_type": db_file.file_type,
                "upload_date": db_file.upload_date,
                "severity_overlay": db_file.severity_overlay,
                "segmentation_mask": db_file.segmentation_mask,
                "heatmap_overlay": db_file.heatmap_overlay,
                "cam_overlay": db_file.cam_overlay,
                "rendering_3d": db_file.rendering_3d,
                "rendering_png": db_file.rendering_png,
                "patient_uid": patientUid,
                "doctor_name": doctor_name,
                "tb_prediction": {
                    "prediction": 0 if tb_prediction['Prediction'] != "TB" else 1,
                    "confidence": tb_prediction['Confidence'],
                    "result": "TB Positive" if tb_prediction['Prediction'] == "TB" else "TB Negative"
                } if tb_prediction else None
            })
            print(db_file.id)
            print(filename)
            print(file_url)
            print(preview_url)
            print(db_file.file_type)
            print(db_file.upload_date)
            print(patientUid)
            print(doctor_name)
            print({
                    "prediction": 0 if tb_prediction['Prediction'] != "TB" else 1,
                    "confidence": tb_prediction['Confidence'],
                    "result": "TB Positive" if tb_prediction['Prediction'] == "TB" else "TB Negative"
                } if tb_prediction else None)
            return {
                "id": db_file.id,
                "filename": filename,
                "file_url": file_url,
                "preview_url": preview_url,
                "file_type": db_file.file_type,
                "upload_date": db_file.upload_date,
                "patient_uid": patientUid,
                "doctor_name": doctor_name,
                "severity_overlay": db_file.severity_overlay,
                "segmentation_mask": db_file.segmentation_mask,
                "heatmap_overlay": db_file.heatmap_overlay,
                "cam_overlay": db_file.cam_overlay,
                "rendering_3d": os.path.splitext(db_file.rendering_3d)[0] + ".vti" if db_file.rendering_3d else None,
                "rendering_png": os.path.splitext(db_file.rendering_png)[0] + ".png" if db_file.rendering_png else None,
                # "tb_prediction": tb_prediction if tb_prediction else None
                "tb_prediction": {
                    "prediction": 0 if tb_prediction['Prediction'] != "TB" else 1,
                    "confidence": tb_prediction['Confidence'],
                    "result": "TB Positive" if tb_prediction['Prediction'] == "TB" else "TB Negative"
                } if tb_prediction else None
            }
            
        except Exception as e:
            db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"Error saving file reference: {str(e)}"
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
    
    # Extract the directory and filename from the stored URL path
    url_parts = file_ref.file_path.strip('/').split('/')
    if len(url_parts) >= 3 and url_parts[0] == "files":
        dir_name = url_parts[1]  # "image" or "dicom"
        filename = url_parts[2]
        file_path = os.path.join(UPLOAD_DIR, dir_name, filename)
    else:
        # Handle legacy file paths
        filename = os.path.basename(file_ref.file_path)
        file_path = os.path.join(UPLOAD_DIR, filename)
        # If not found, try the image directory
        if not os.path.exists(file_path):
            file_path = os.path.join(IMAGE_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found on disk")
    
    # Determine content type based on file extension
    content_type = None
    if file_path.lower().endswith(('.jpg', '.jpeg')):
        content_type = "image/jpeg"
    elif file_path.lower().endswith('.png'):
        content_type = "image/png"
    elif file_path.lower().endswith('.dcm'):
        content_type = "application/dicom"
    elif file_path.lower().endswith('.pdf'):
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
    print(str(files) + "files of patient " + uid)
    return files 

@router.post("/files/render")
async def render_vti_file(
    vti_path: str = Form(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)):
    """
    Load and render a VTI file
    
    Args:
        vti_path (str): Path to the VTI file or URL
        off_screen (bool): If True, rendering is done off-screen
        screenshot (str): Path to save a screenshot, if provided
    """
    # vti_path = request.vti_path
    # Check if it's a URL or local file
    vti_path = os.path.join(UPLOAD_DIR, vti_path.split("files/")[1])
    print(vti_path)
    off_screen = False
    screenshot = None
    try:
        grid = pv.read(vti_path)
        print(f"VTI loaded successfully!")
        print(f"  Dimensions: {grid.dimensions}")
        print(f"  Bounds: {grid.bounds}")
        
        # Get available arrays/fields in the VTI file
        arrays = list(grid.point_data.keys())
        print(f"  Available arrays: {arrays}")
        
        # Create a visualization plotter
        pl = pv.Plotter(off_screen=off_screen)
        
        # If we have XRay and TB arrays, use the standard visualization
        if 'XRay' in arrays and 'TB' in arrays:
            print("Detected XRay and TB arrays - setting up TB visualization")
            pl.add_volume(grid, scalars='XRay', opacity=[0, 0.8], cmap='gray', clim=[0, 1])
            pl.add_volume(grid, scalars='TB', opacity=[0, 0.5, 0.9], cmap='hot', clim=[0, 1])
        # If not, try to render any scalar arrays present
        elif len(arrays) > 0:
            primary_array = arrays[0]
            print(f"Using primary array '{primary_array}' for visualization")
            pl.add_volume(grid, scalars=primary_array, cmap='viridis')
        else:
            # If no scalar arrays, render as a simple outline
            print("No scalar arrays found. Rendering outline only.")
            pl.add_mesh(grid.outline(), color='white')
        
        
        # Add a custom text with file info
        pl.add_text(f"File: {os.path.basename(vti_path)}\nDims: {grid.dimensions}", font_size=12)
        # Take a screenshot if requested
        if screenshot:
            pl.screenshot(screenshot)
            print(f"Screenshot saved to: {screenshot}")
        
        if not off_screen:
            print("\nInteraction Controls:")
            print("  Mouse Left: Rotate")
            print("  Mouse Middle/Wheel: Zoom")
            print("  Mouse Right: Pan")
            print("  'r' key: Reset view")
            print("  'q' key: Quit")
            print("\nRendering VTI file...")
            pl.show()
        else:
            pl.close()
        
        return True
    
    except Exception as e:
        print(f"Error loading or rendering VTI file: {e}")
        return False