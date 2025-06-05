from pydantic_settings import BaseSettings
from typing import List
import os
import sys
from dotenv import load_dotenv
from pathlib import Path

# Determine base directory for both development and packaged environments
if getattr(sys, 'frozen', False):
    # We are running in a bundled app
    # Use a fixed location for persistent data
    BASE_DIR = Path(os.path.expanduser("~")) / ".pivotal"
    # Create the directory if it doesn't exist
    os.makedirs(BASE_DIR, exist_ok=True)
else:
    # We are running in a normal Python environment
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # API Settings
    # API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Pivotal TB Screening"
    
    # Load environment variables
    load_dotenv()
    
    # Database Settings
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'pivotal.db'}"
    
    # JWT Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-for-jwt")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # File Upload Settings
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "dcm"]
    
    # Optional OTP Settings
    OTP_ENABLED: bool = False
    OTP_EXPIRY_MINUTES: int = 10
    SMS_API_KEY: str = os.getenv("SMS_API_KEY", "")

    class Config:
        case_sensitive = True

settings = Settings() 