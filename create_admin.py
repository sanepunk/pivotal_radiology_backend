import sys
import os
from datetime import datetime
from passlib.context import CryptContext
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from pathlib import Path

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import app models
from app.models.user import User
from app.core.database import Base, engine

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_admin_user():
    # Create database and tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Create session
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    
    try:
        # Check if admin user already exists
        existing_admin = db.query(User).filter(User.email == "admin@pivotal.com").first()
        
        if existing_admin:
            print("Admin user already exists.")
            return
        
        # Create admin user
        admin_user = User(
            email="admin@pivotal.com",
            name="Admin",
            role="admin",
            hashed_password=get_password_hash("admin123"),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            is_active=True
        )
        
        db.add(admin_user)
        db.commit()
        
        print("Admin user created successfully!")
        print("Email: admin@pivotal.com")
        print("Password: admin123")
        print("\nPlease change the password after first login.")
        
    except Exception as e:
        db.rollback()
        print(f"Error creating admin user: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    create_admin_user() 