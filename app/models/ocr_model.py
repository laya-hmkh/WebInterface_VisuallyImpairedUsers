import easyocr
import logging
import os
from typing import Tuple, List, Dict, Union

# Configure logging for debugging and status updates
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize EasyOCR reader for English text extraction (CPU-only mode)
try:
    reader = easyocr.Reader(["en"], gpu=False)
    logger.info("EasyOCR Reader initialized successfully with English language support (GPU disabled).")
except Exception as e:
    logger.error(f"Failed to initialize EasyOCR Reader: {str(e)}")
    raise

def extract_text_from_image(image_path):
    """
    Extract text from an image using EasyOCR.

    Args:
        image_path (str): Path to the input image file.

    Returns:
        str: Extracted text joined by spaces, or "No text detected." if none found.

    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If text extraction fails due to processing errors.
        ValueError: If image_path is not a string.
    """
    # Validate image_path type
    if not isinstance(image_path, str):
        logger.error(f"Invalid image_path type: {type(image_path)}. Expected string.")
        raise ValueError("image_path must be a string.")
    
    # Check if image file exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        logger.info(f"Extracting text from image: {image_path}")
        # Extract text using EasyOCR (detail=0 for text only, no bounding boxes)
        text = reader.readtext(image_path, detail=0)
        
        # Combine extracted text or return default message if empty
        result = ' '.join(text) if text else "No text detected."
        logger.info(f"Extracted text: {result}")
        return result

    except Exception as e:
        logger.error(f"Error during text extraction: {str(e)}")
        raise RuntimeError(f"Failed to extract text from image: {str(e)}")