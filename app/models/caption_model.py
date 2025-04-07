import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for TensorFlow
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import logging

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Select device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load the BLIP model and processor for image captioning
try:
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    logger.info("Model and processor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model components: {str(e)}")
    raise

def generate_caption(image):
    """
    Generate a caption for an input image using the BLIP model.

    Args:
        image (PIL.Image.Image): Input image to caption.

    Returns:
        str: Generated caption for the image.

    Raises:
        ValueError: If input is not a valid PIL Image.
        RuntimeError: If caption generation encounters an error.
    """
    # Validate input image type
    if not isinstance(image, Image.Image):
        logger.error("Invalid image input received.")
        raise ValueError("Invalid image input. Expected a PIL Image object.")

    try:
        logger.info("Processing image for caption generation...")
        # Prepare inputs for conditional image captioning with a starter text
        text = "The image of"
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        
        # Generate caption using beam search for improved quality
        logger.info("Generating caption...")
        with torch.no_grad():  # Disable gradients for faster inference
            outputs = model.generate(**inputs, num_beams=4)  # Beam search with 4 beams
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Generated caption (length: {len(caption)} chars)")
        return caption

    except Exception as e:
        logger.error(f"Error during caption generation: {str(e)}")
        raise RuntimeError(f"Failed to generate caption: {str(e)}")