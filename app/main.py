import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for TensorFlow compatibility
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from models.caption_model import generate_caption  # Import image captioning function
from models.ocr_model import extract_text_from_image  # Import OCR text extraction function
from models.tts_model import text_to_speech  # Import text-to-speech function
from PIL import Image
import io
import traceback
import uuid
from datetime import datetime
import logging
import threading

from io import BytesIO
from PIL import Image

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define base directories for templates and static files
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.abspath(os.path.join(BASE_DIR, '../templates'))
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, '../static'))

# Ensure directories exist for serving static files and templates
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Initialize Flask app with custom template and static folders
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

@app.before_request
def intercept_favicon():
    """Prevent favicon requests from triggering errors by returning a no-content response."""
    if request.path == '/favicon.ico':
        return Response(status=204)  # HTTP 204 No Content

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files (e.g., audio files) from the static directory."""
    file_path = os.path.join(STATIC_DIR, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(STATIC_DIR, filename)

@app.route('/')
def index():
    """Render the main HTML page for the web interface."""
    logger.info(f"Template directory path: {TEMPLATES_DIR}")
    return render_template('index.html')

@app.route('/caption', methods=['GET'])
def caption_image():
    """
    Process an image URL to generate a caption, extract text via OCR, and synthesize audio.

    Returns:
        JSON response with caption, OCR text, and audio URL, or an error message.
    """
    image_url = request.args.get("image_url")

    # Validate input URL
    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400

    try:
        logger.info(f"Fetching image from URL: {image_url}")

        # Download and open image from URL
        image = Image.open(requests.get(image_url, stream=True, timeout=10).raw).convert('RGB')

        # Save temporary image for OCR processing
        image_path = os.path.join(STATIC_DIR, "temp_image.jpg")
        image.save(image_path)
        
        logger.info("Running OCR and Image Captioning simultaneously...")

        # Dictionaries to store results from threads
        caption_result = {}
        ocr_result = {}
        
        def run_caption():
            """Thread function to generate image caption."""
            try:
                caption = generate_caption(image)
                caption_result['caption'] = caption
            except Exception as e:
                caption_result['error'] = str(e)
        
        def run_ocr():
            """Thread function to extract text from image."""
            try:
                extracted_text = extract_text_from_image(image_path)
                ocr_result['text'] = extracted_text
            except Exception as e:
                ocr_result['error'] = str(e)
        
        # Execute captioning and OCR in parallel using threads
        caption_thread = threading.Thread(target=run_caption)
        ocr_thread = threading.Thread(target=run_ocr)
        
        caption_thread.start()
        ocr_thread.start()
        
        caption_thread.join()
        ocr_thread.join()
        
        # Check for thread execution errors
        if 'error' in caption_result:
            raise Exception(f"Captioning failed: {caption_result['error']}")
        if 'error' in ocr_result:
            raise Exception(f"OCR failed: {ocr_result['error']}")
        
        caption = caption_result.get('caption', 'No caption generated')
        extracted_text = ocr_result.get('text', 'No text detected')

        logger.info(f"Generated caption: {caption}")
        logger.info(f"Extracted OCR text: {extracted_text}")

        # Combine caption and OCR text for TTS
        final_text = f"Image captioning content is: {caption}. And text in the image is: {extracted_text}."
        logger.info(f"Final text for TTS: {final_text}")
        
        # Generate unique audio filename with timestamp
        unique_filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        audio_path = os.path.join(STATIC_DIR, unique_filename)

        # Generate audio from combined text
        logger.info("Running TTS...")
        generated_audio = text_to_speech(final_text, output_path=audio_path)

        if not generated_audio or not os.path.exists(audio_path):
            logger.error("Audio file was not created!")
            return jsonify({'error': 'Audio file not found'}), 500

        logger.info(f"Audio file saved at: {audio_path}")

        # Return results as JSON
        return jsonify({
            'caption': caption,
            'ocr_text': extracted_text,
            'audio_url': f"/static/{unique_filename}"
        })

    except requests.RequestException as e:
        logger.error(f"Error fetching image: {str(e)}")
        return jsonify({'error': f"Failed to fetch image: {str(e)}"}), 500
    except Exception as e:
        logger.error("Server error:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/tts', methods=['GET'])
def text_to_speech_api():
    """
    Convert provided text to speech and return the audio file URL.

    Returns:
        JSON response with audio URL or an error message.
    """
    text = request.args.get("text")

    # Validate input text
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        logger.info(f"Converting text to speech: {text}")

        # Generate unique audio filename with timestamp
        unique_filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        audio_path = os.path.join(STATIC_DIR, unique_filename)

        # Convert text to speech
        generated_audio = text_to_speech(text, output_path=audio_path)

        if not generated_audio or not os.path.exists(audio_path):
            logger.error("Audio file was not created!")
            return jsonify({'error': 'Audio file not found'}), 500

        logger.info(f"Audio file saved at: {audio_path}")

        # Return audio URL as JSON
        return jsonify({'audio_url': f"/static/{unique_filename}"})

    except Exception as e:
        logger.error("Server error:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app in debug mode on port 5000, accessible from any host
    app.run(debug=True, port=5000, host='0.0.0.0')