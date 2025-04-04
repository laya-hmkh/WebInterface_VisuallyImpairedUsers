import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import requests
from flask import Flask, request, jsonify, render_template, send_from_directory
from models.caption_model import generate_caption
from models.ocr_model import extract_text_from_image
from models.tts_model import text_to_speech
from PIL import Image
import io
import traceback
import uuid
from datetime import datetime
import logging
import threading

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up base directories
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATES_DIR = os.path.abspath(os.path.join(BASE_DIR, '../templates'))
STATIC_DIR = os.path.abspath(os.path.join(BASE_DIR, '../static'))

# Ensure directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    file_path = os.path.join(STATIC_DIR, filename)
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(STATIC_DIR, filename)

@app.route('/')
def index():
    logger.info(f"Template directory path: {TEMPLATES_DIR}")
    return render_template('index.html')

@app.route('/caption', methods=['GET'])
def caption_image():
    image_url = request.args.get("image_url")

    if not image_url:
        return jsonify({'error': 'No image URL provided'}), 400

    try:
        logger.info(f"Fetching image from URL: {image_url}")

        # Download image from URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Save temporary image for OCR
        image_path = os.path.join(STATIC_DIR, "temp_image.jpg")
        image.save(image_path)
        
        logger.info("Running OCR and Image Captioning simultaneously...")

        caption_result = {}
        ocr_result = {}
        
        def run_caption():
            try:
                caption = generate_caption(image)
                caption_result['caption'] = caption
            except Exception as e:
                caption_result['error'] = str(e)
        
        def run_ocr():
            try:
                extracted_text = extract_text_from_image(image_path)
                ocr_result['text'] = extracted_text
            except Exception as e:
                ocr_result['error'] = str(e)
        
        # Create and start threads
        caption_thread = threading.Thread(target=run_caption)
        ocr_thread = threading.Thread(target=run_ocr)
        
        caption_thread.start()
        ocr_thread.start()
        
        # Wait for both threads to complete
        caption_thread.join()
        ocr_thread.join()
        
        # Check for errors
        if 'error' in caption_result:
            raise Exception(f"Captioning failed: {caption_result['error']}")
        if 'error' in ocr_result:
            raise Exception(f"OCR failed: {ocr_result['error']}")
        
        caption = caption_result.get('caption', 'No caption generated')
        extracted_text = ocr_result.get('text', 'No text detected')

        logger.info(f"Generated caption: {caption}")
        logger.info(f"Extracted OCR text: {extracted_text}")

        # Rest of the code remains the same...
        # Combine texts for TTS
        final_text = f"Image captioning content: {caption}. And OCR content: {extracted_text}."
        logger.info(f"Final text for TTS: {final_text}")
        
        # Generate a unique filename for the audio file
        unique_filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        audio_path = os.path.join(STATIC_DIR, unique_filename)

        # Convert text to speech and save audio file
        logger.info("Running TTS...")
        generated_audio = text_to_speech(final_text, output_path=audio_path)

        if not generated_audio or not os.path.exists(audio_path):
            logger.error("Audio file was not created!")
            return jsonify({'error': 'Audio file not found'}), 500

        logger.info(f"Audio file saved at: {audio_path}")

        return jsonify({
            'caption': caption,
            'ocr_text': extracted_text,
            'audio_url': f"/static/{unique_filename}"
        })

        # # Perform Image Captioning
        # logger.info("Running Image Captioning...")
        # caption = generate_caption(image)
        # logger.info(f"Generated caption: {caption}")

        # # Perform OCR
        # logger.info("Running OCR...")
        # extracted_text = extract_text_from_image(image_path)
        # logger.info(f"Extracted OCR text: {extracted_text}")

    except requests.RequestException as e:
        logger.error(f"Error fetching image: {str(e)}")
        return jsonify({'error': f"Failed to fetch image: {str(e)}"}), 500
    except Exception as e:
        logger.error("Server error:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/tts', methods=['GET'])
def text_to_speech_api():
    text = request.args.get("text")

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        logger.info(f"Converting text to speech: {text}")

        # Generate a unique filename for the audio file
        unique_filename = f"{uuid.uuid4().hex}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        audio_path = os.path.join(STATIC_DIR, unique_filename)

        # Convert text to speech and save audio file
        generated_audio = text_to_speech(text, output_path=audio_path)

        if not generated_audio or not os.path.exists(audio_path):
            logger.error("Audio file was not created!")
            return jsonify({'error': 'Audio file not found'}), 500

        logger.info(f"Audio file saved at: {audio_path}")

        return jsonify({'audio_url': f"/static/{unique_filename}"})

    except Exception as e:
        logger.error("Server error:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')