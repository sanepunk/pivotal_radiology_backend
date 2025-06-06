import sys
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
from dotenv import load_dotenv
from pathlib import Path

from app.core.config import settings
from app.api.routes import auth, patients, analysis
from app.core.database import Base, engine
import create_admin

# Load environment variables
load_dotenv()

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title="Pivotal TB Screening API",
    description="Backend API for TB Screening Application",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine base directory for file storage
# Handle both development and packaged environments
if getattr(sys, 'frozen', False):
    # Running as compiled executable
    base_dir = Path(sys.executable).parent
else:
    # Running in development environment
    base_dir = Path(__file__).resolve().parent

# Create uploads directory if it doesn't exist
UPLOAD_DIR = os.path.join(base_dir, 'uploads')
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)
    print(f"Created uploads directory at: {UPLOAD_DIR}")
else:
    print(f"Using existing uploads directory at: {UPLOAD_DIR}")

# Make sure the database is in the correct location
if getattr(sys, 'frozen', False):
    # For packaged app, use a fixed location in user's home directory
    HOME_DIR = os.path.expanduser("~")
    PERSISTENT_DIR = os.path.join(HOME_DIR, '.pivotal')
    os.makedirs(PERSISTENT_DIR, exist_ok=True)
    DB_PATH = os.path.join(PERSISTENT_DIR, 'pivotal.db')
    print(f"Using persistent database at: {DB_PATH}")
else:
    DB_PATH = os.path.join(base_dir, 'pivotal.db')

DEV_DB_PATH = os.path.join(Path(__file__).resolve().parent, 'pivotal.db')

# If the DB doesn't exist, copy it from dev location
if not os.path.exists(DB_PATH) and os.path.exists(DEV_DB_PATH):
    shutil.copy2(DEV_DB_PATH, DB_PATH)
    print(f"Copied database from {DEV_DB_PATH} to {DB_PATH}")

# Location of frontend files - handle both development and packaged environments
if getattr(sys, 'frozen', False):
    # For packaged app, get path from PyInstaller bundle
    import sys
    import os
    
    # This is the PyInstaller way to find bundled data
    def resource_path(relative_path):
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return os.path.join(base_path, relative_path)
    
    FRONTEND_DIST = resource_path('dist1')
    print(f"Using frontend files from PyInstaller bundle: {FRONTEND_DIST}")
else:
    # In development
    FRONTEND_DIST = os.path.join(base_dir, 'dist1')
    print(f"Using frontend files from: {FRONTEND_DIST}")

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = """
        <!DOCTYPE html>
    <html>
        <head>
            <meta http-equiv="refresh" content="0;url=/app" />
            <title>Redirecting...</title>
        </head>
        <body>
            <p>Redirecting to application...</p>
            <script>window.location.href = "/app";</script>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(patients.router, prefix="/patients", tags=["Patients"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])

@app.exception_handler(404)
async def spa_fallback(request: Request, exc):
    if request.url.path.startswith("/app"):
        try:
            with open(os.path.join(FRONTEND_DIST, "index.html")) as f:
                return HTMLResponse(content=f.read())
        except FileNotFoundError:
            return JSONResponse(
                status_code=404, 
                content={"detail": f"Frontend files not found at {FRONTEND_DIST}"}
            )
    return JSONResponse(status_code=404, content={"detail": "Not Found"})

# Check if the uploads and frontend directories exist before mounting
# For packaged app, use the persistent uploads directory
if getattr(sys, 'frozen', False):
    HOME_DIR = os.path.expanduser("~")
    PERSISTENT_DIR = os.path.join(HOME_DIR, '.pivotal')
    PERSISTENT_UPLOAD_DIR = os.path.join(PERSISTENT_DIR, 'uploads')
    
    # Make sure directories exist
    os.makedirs(PERSISTENT_UPLOAD_DIR, exist_ok=True)
    
    # Create image and dicom subdirectories if they don't exist
    PERSISTENT_IMAGE_DIR = os.path.join(PERSISTENT_UPLOAD_DIR, 'image')
    PERSISTENT_DICOM_DIR = os.path.join(PERSISTENT_UPLOAD_DIR, 'dicom')
    os.makedirs(PERSISTENT_IMAGE_DIR, exist_ok=True)
    os.makedirs(PERSISTENT_DICOM_DIR, exist_ok=True)
    
    print(f"Mounting files from persistent uploads directory: {PERSISTENT_UPLOAD_DIR}")
    
    # Mount the uploads directory and its subdirectories
    app.mount("/files", StaticFiles(directory=PERSISTENT_UPLOAD_DIR), name="files")
elif os.path.exists(UPLOAD_DIR):
    # In development mode, create subdirectories if they don't exist
    IMAGE_DIR = os.path.join(UPLOAD_DIR, 'image')
    DICOM_DIR = os.path.join(UPLOAD_DIR, 'dicom')
    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(DICOM_DIR, exist_ok=True)
    
    # Mount the uploads directory
    app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")
else:
    print(f"WARNING: Uploads directory does not exist: {UPLOAD_DIR}")

if os.path.exists(FRONTEND_DIST):
    app.mount("/app", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:
    print(f"WARNING: Frontend directory does not exist: {FRONTEND_DIST}")

import uvicorn

if __name__ == "__main__":
    import argparse
    import socket
    
    # Default port to try first
    default_port = 8000
    
    # Parse command line arguments if provided
    parser = argparse.ArgumentParser(description="Start the TB Screening Application server")
    parser.add_argument('--port', type=int, default=default_port, help='Port to run the server on')
    args = parser.parse_args()
    
    # Try the specified port, if busy try alternatives (8000, 8080, 8888, 9000)
    ports_to_try = [args.port]
    if args.port == default_port:
        ports_to_try.extend([])
    
    for port in ports_to_try:
        try:
            # Quick check if port is in use
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("127.0.0.1", port))
            s.close()
            
            # Port is available, run the app
            print(f"Starting server on port {port}")
            print(f"Open your browser to http://localhost:{port}")
            uvicorn.run(app, host="127.0.0.1", port=port, workers=1)
            break
        except OSError:
            print(f"Port {port} is busy, trying next port...")
            continue
    else:
        print("All ports are busy. Please close some applications and try again.")
        import time
        time.sleep(5)