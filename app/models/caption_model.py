import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model and components
try:
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast = True)
    logger.info("Model and processor loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model components: {str(e)}")
    raise

def generate_caption(image):
    """
    Generate a caption for a given image using a pre-trained VisionEncoderDecoderModel.

    Args:
        image (PIL.Image.Image): The input image to caption.

    Returns:
        str: The generated caption.

    Raises:
        ValueError: If the input is not a valid PIL Image.
        RuntimeError: If caption generation fails.
    """
    if not isinstance(image, Image.Image):
        logger.error("Invalid image input received.")
        raise ValueError("Invalid image input. Expected a PIL Image object.")

    try:
        logger.info("Processing image for caption generation...")
        # conditional image captioning
        text = "The image of"
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        
        # Generate caption
        logger.info("Generating caption...")
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model.generate(**inputs, num_beams=4)  # Added parameters for better control
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        
        logger.info(f"Generated caption (length: {len(caption)} chars)")
        return caption

    except Exception as e:
        logger.error(f"Error during caption generation: {str(e)}")
        raise RuntimeError(f"Failed to generate caption: {str(e)}")