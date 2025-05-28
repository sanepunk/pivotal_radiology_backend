from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from motor.motor_asyncio import AsyncIOMotorClient
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os

from app.core.config import settings
from app.api.routes import auth, patients, analysis
# from app.core.database import connect_to_mongo, close_mongo_connection

# Load environment variables
load_dotenv()

# MongoDB connection
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Connect to MongoDB
    app.mongodb_client = AsyncIOMotorClient(settings.MONGODB_URL)
    app.mongodb = app.mongodb_client[settings.MONGODB_NAME]
    print("Connected to MongoDB")
    yield
    # Close MongoDB connection
    app.mongodb_client.close()
    print("Closed MongoDB connection")

# Create FastAPI app
app = FastAPI(
    title="Pivotal TB Screening API",
    description="Backend API for TB Screening Application",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://teleradio.netlify.app"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"  # Keep it simple with a relative path
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# Mount the uploads directory
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")

# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(patients.router, prefix="/patients", tags=["Patients"])
app.include_router(analysis.router, prefix="/analysis", tags=["Analysis"])

# Database connection events
# app.add_event_handler("startup", connect_to_mongo)
# app.add_event_handler("shutdown", close_mongo_connection)

@app.get("/")
async def root():
    return {"message": "Welcome to Pivotal TB Screening API"}