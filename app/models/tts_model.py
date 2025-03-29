import torch
import soundfile as sf
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load TTS model from Hugging Face
logger.info("Loading TTS model...")
try:
    model_speech, symbols, sample_rate, example_text, apply_tts = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='en',
        speaker='lj_16khz'
    )
    # Set model to CPU
    model = model_speech.to("cpu")
    logger.info("TTS model loaded and ready.")
except Exception as e:
    logger.error(f"Failed to load TTS model: {str(e)}")
    raise

def text_to_speech(text, output_path="static/output.wav"):
    """
    Convert text to speech and save it to the specified path.

    Args:
        text (str): The text to convert to speech.
        output_path (str, optional): Path where the audio file will be saved. Defaults to "static/output.wav".

    Returns:
        str or None: The path to the saved audio file if successful, None otherwise.

    Raises:
        ValueError: If the text is not a valid string.
        RuntimeError: If audio generation or saving fails.
    """
    if not isinstance(text, str) or not text.strip():
        logger.error("Invalid text input: Must be a non-empty string.")
        raise ValueError("Text must be a non-empty string.")

    logger.info(f"Converting text to speech: {text}")

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Generate audio
        audio = apply_tts(
            texts=[text],
            model=model,
            sample_rate=sample_rate,
            symbols=symbols,
            device="cpu"
        )[0].numpy()

        # Save audio file
        sf.write(output_path, audio, sample_rate)
        logger.info(f"Audio file saved at: {output_path}")
        return output_path

    except RuntimeError as e:
        logger.error(f"Error during TTS execution: {str(e)}")
        
        # Handle specific tensor size mismatch error
        if "Sizes of tensors must match" in str(e):
            logger.warning("Attempting to handle attention_weights tensor size mismatch...")
            return None
        
        return None

    except Exception as e:
        logger.error(f"Unexpected error during TTS: {str(e)}")
        raise RuntimeError(f"Failed to generate or save audio: {str(e)}")