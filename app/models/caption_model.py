from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast
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
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning").to(device)
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    logger.info("Model, processor, and tokenizer loaded successfully.")
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
        # Prepare image for the model
        inputs = processor(images=image, return_tensors="pt").to(device)
        
        # Generate caption
        logger.info("Generating caption...")
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model.generate(**inputs, max_length=16, num_beams=4)  # Added parameters for better control
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        logger.info(f"Generated caption: {caption}")
        return caption

    except Exception as e:
        logger.error(f"Error during caption generation: {str(e)}")
        raise RuntimeError(f"Failed to generate caption: {str(e)}")