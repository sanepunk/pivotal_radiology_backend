# Pivotal TB Screening Backend

This is the backend API for the TB Screening application, now using SQLite for easier development and deployment.

## Setup Instructions

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Installation

1. Create and activate a virtual environment (optional but recommended):
```
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Create admin user:
```
python create_admin.py
```
This will create an admin user with the following credentials:
- Email: admin@pivotal.com
- Password: admin123

### Running the Application

Start the FastAPI server:
```
uvicorn main:app --reload
```

The API will be available at:
- API: http://localhost:8000
- Swagger Documentation: http://localhost:8000/docs
- ReDoc Documentation: http://localhost:8000/redoc
- Frontend: http://localhost:8000/app

### Database

The application now uses SQLite instead of MongoDB. The database file is created automatically at:
```
/pivotal.db
```

All data is stored in this file, which makes development and deployment easier as you don't need to set up a separate database server.

### Authentication

The API uses JWT bearer tokens for authentication. To authenticate:

1. Login at `/auth/login` with admin credentials or create a new user at `/auth/register`
2. Use the returned token in the Authorization header for subsequent requests:
```
Authorization: Bearer {your_token}
```

### API Structure

- `/auth` - Authentication endpoints (login, register, verify token)
- `/patients` - Patient management endpoints
- `/analysis` - Image analysis endpoints

## Development

The backend is built with:

- FastAPI for the API framework
- SQLAlchemy for ORM
- SQLite for the database
- JWT for authentication

## Folder Structure

- `app/` - Main application directory
  - `api/` - API routes
    - `routes/` - Route modules
  - `core/` - Core functionality
    - `config.py` - Application configuration
    - `database.py` - Database setup
  - `models/` - Data models
- `uploads/` - Uploaded files
- `main.py` - Application entry point
- `create_admin.py` - Script to create admin user

## Packaging and Deployment

### Prerequisites for Packaging
- PyInstaller: `pip install pyinstaller`
- Frontend build in the `dist1` directory

### Preparing Frontend Files
1. Build your frontend application
2. Copy all frontend build files to the `dist1` directory in this project

### Packaging the Application
We provide two ways to package the application:

#### Method 1: Using the package.py script
```
python package.py
```
This script will:
- Check for required directories
- Create them if needed
- Clean up previous build artifacts
- Run PyInstaller with the correct configuration

#### Method 2: Using PyInstaller directly
```
pyinstaller main.spec
```

### Deployment
The packaged application will be in the `dist` directory. To deploy:

1. Copy the entire `dist` directory to the target machine
2. Run the application by executing `main.exe` or using the included batch file:
   ```
   run_pivotal.bat
   ```

### Important Notes for Deployment
- The application will create an `uploads` directory in the same folder as the executable if it doesn't exist
- The database file `pivotal.db` will be created or used from the same directory as the executable
- All files and database will persist between runs of the application
- If you need to update existing file paths in the database for compatibility with the packaged version, run:
  ```
  python update_file_paths.py
  ```

## Troubleshooting
- **Missing Frontend Files**: Make sure the `dist1` directory contains all frontend build files
- **Database Errors**: If you encounter database errors, delete the `pivotal.db` file and restart the application
- **Missing Files**: Ensure the `uploads` directory exists in the same folder as the executable
- **Image Not Found Errors**: Run the `update_file_paths.py` script to update file paths in the database 