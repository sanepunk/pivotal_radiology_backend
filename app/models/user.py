from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal
from datetime import datetime

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    email: Optional[str] = None

class UserBase(BaseModel):
    email: EmailStr = Field(..., description="User email")
    name: str = Field(..., description="Full name of the user")
    role: str = Field(..., description="User role", pattern="^(doctor|admin)$")

class UserCreate(UserBase):
    password: str = Field(..., min_length=6, description="User password")
    confirmPassword: str = Field(..., description="Confirm password")

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserInDB(UserBase):
    hashed_password: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    class Config:
        from_attributes = True 