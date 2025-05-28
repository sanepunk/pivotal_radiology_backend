from pydantic_settings import BaseSettings
from typing import List
import os
from dotenv import load_dotenv

class Settings(BaseSettings):
    # API Settings
    # API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Pivotal TB Screening"
    load_dotenv()
    # MongoDB Settings
    MONGODB_URL: str = os.getenv("MONGODB_URL")
    MONGODB_NAME: str = os.getenv("MONGODB_NAME", "pivotal")
    
    # JWT Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-for-jwt")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # File Upload Settings
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = ["jpg", "jpeg", "png", "dcm"]
    
    # Optional OTP Settings
    OTP_ENABLED: bool = False
    OTP_EXPIRY_MINUTES: int = 10
    SMS_API_KEY: str = os.getenv("SMS_API_KEY", "")

    class Config:
        case_sensitive = True

settings = Settings() 