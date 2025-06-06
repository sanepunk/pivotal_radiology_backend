import os
import pydicom
import io
from PIL import Image
import numpy as np

def convert_dicom_to_png(dicom_content, patient_uid):
    """
    Convert a DICOM file content to a PNG image using pydicom and pillow.
    
    Args:
        dicom_content (bytes): The raw content of the DICOM file
        patient_uid (str): The patient UID for filename
    
    Returns:
        tuple: (png_data as bytes, original_filename)
    """
    try:
        # Create a file-like object from the content
        buffer = io.BytesIO(dicom_content)
        
        # Read the DICOM dataset
        ds = pydicom.dcmread(buffer)
        
        # Check if the file contains pixel data
        if 'PixelData' not in ds:
            raise ValueError("No pixel data found in the DICOM file.")
        
        # Extract the image data
        pixel_array = ds.pixel_array
        
        # DICOM pixel data often needs to be normalized
        if pixel_array.dtype != np.uint8:
            # Scale to 8-bit range (0-255)
            if pixel_array.max() > 0:
                # Normalize based on data range
                pixel_array = ((pixel_array - pixel_array.min()) / 
                              (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            else:
                pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        
        # Convert numpy array to PIL Image
        if len(pixel_array.shape) == 3 and pixel_array.shape[2] == 3:  # RGB
            mode = "RGB"
        else:  # Grayscale
            mode = "L"
        
        img = Image.fromarray(pixel_array, mode=mode)
        
        # Save the image to bytes buffer
        output_buffer = io.BytesIO()
        img.save(output_buffer, format="PNG")
        output_buffer.seek(0)
        
        # Get patient ID and modality metadata if available
        patient_id = ds.get('PatientID', 'Unknown')
        modality = ds.get('Modality', 'Unknown')
        
        # Create a descriptive filename that matches the format in patients.py
        # The actual filename will be set in the patients.py route with the timestamp
        original_filename = f"{patient_uid}.png"
        
        # Store metadata for potential use
        metadata = {
            'patient_id': patient_id,
            'modality': modality
        }
        
        return output_buffer.getvalue(), original_filename, metadata
        
    except Exception as e:
        raise Exception(f"Error converting DICOM to PNG: {str(e)}") 