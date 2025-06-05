#!/usr/bin/env python
import os
import subprocess
import shutil
from pathlib import Path

def main():
    # Get the current directory
    base_dir = Path(__file__).parent.absolute()
    
    # Ensure we have all necessary directories
    print("Checking directories...")
    
    # Check for dist1 (frontend files) directory
    dist1_dir = os.path.join(base_dir, 'dist1')
    if not os.path.exists(dist1_dir):
        os.makedirs(dist1_dir, exist_ok=True)
        print(f"Created dist1 directory at {dist1_dir}")
        print("WARNING: dist1 directory is empty. Please copy frontend files to dist1 directory.")
    else:
        print(f"Using existing dist1 directory at {dist1_dir}")
        if not os.listdir(dist1_dir):
            print("WARNING: dist1 directory is empty. Please copy frontend files to dist1 directory.")
    
    # Check for uploads directory
    uploads_dir = os.path.join(base_dir, 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir, exist_ok=True)
        print(f"Created uploads directory at {uploads_dir}")
    else:
        print(f"Using existing uploads directory at {uploads_dir}")
    
    # Check for database
    db_path = os.path.join(base_dir, 'pivotal.db')
    if not os.path.exists(db_path):
        print(f"WARNING: Database file not found at {db_path}. The packaged app will create a new database.")
    else:
        print(f"Using existing database at {db_path}")
    
    # Clean up previous build artifacts
    print("\nCleaning up previous build artifacts...")
    dist_dir = os.path.join(base_dir, 'dist')
    build_dir = os.path.join(base_dir, 'build')
    
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
        print(f"Removed {dist_dir}")
    
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        print(f"Removed {build_dir}")
    
    # Run PyInstaller
    print("\nPackaging the application with PyInstaller...")
    result = subprocess.run(["pyinstaller", "main.spec"], cwd=base_dir, check=False)
    
    if result.returncode != 0:
        print(f"PyInstaller exited with error code {result.returncode}")
        return
    
    print("\nPackaging complete.")
    print(f"The packaged application is located at: {os.path.join(dist_dir, 'main.exe')}")
    print("\nTo run the application:")
    print(f"  1. Copy the entire {dist_dir} folder to the target machine")
    print("  2. Run main.exe")
    print("\nNote: The database and uploads directory will be preserved between runs.")

if __name__ == "__main__":
    main() 