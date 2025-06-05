from sqlalchemy import Column, Integer, String, Boolean, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal, List

from app.core.database import Base

# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    role = Column(String, nullable=False)  # "doctor" or "admin"
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

# Pydantic Models for API Validation and Response
class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    email: Optional[str] = None

class UserBase(BaseModel):
    email: EmailStr = Field(..., description="User email")
    name: str = Field(..., description="Full name of the user")
    role: str = Field(..., description="User role", pattern="^(doctor|admin)$")

    class Config:
        from_attributes = True

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="User password")
    confirmPassword: str = Field(..., description="Confirm password")

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserInDB(UserBase):
    id: int
    hashed_password: str
    created_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

class UserResponse(UserBase):
    id: int
    created_at: datetime
    
    class Config:
        from_attributes = True 