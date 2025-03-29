Real-Time Image Interpretation and Speech Generation via Web Interfaces for Visually Impaired User


# Image Captioning, OCR, and TTS Web App

This project is a Flask-based web application that combines image captioning, optical character recognition (OCR), and text-to-speech (TTS) functionalities. Users can hover over images to generate captions and OCR text, or over text snippets to hear them spoken aloud.

## Features
- **Image Captioning**: Generates captions for images using a Vision Transformer (ViT) and GPT-2 model.
- **OCR**: Extracts text from images using EasyOCR.
- **TTS**: Converts text to speech using Silero TTS models.
- **Interactive UI**: Hover-based interactions with a modern, responsive design.

## Prerequisites
- Python 3.8+
- Internet connection (for initial model downloads)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ImageCaption-OCR-TTS.git
   cd ImageCaption-OCR-TTS
   ```
2. Create a virtual environment and activate it:
   ```
   python -m venv webIn
   webIn\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the application:
  ```
  python main.py
  ```
5. Open you browser and navigate to: http://localhost:5000

## Usage
- Hover over the sample image to see its caption and OCR text, and hear the combined result.
- Hover over text snippets to hear them spoken aloud.
- The audio player at the bottom plays the generated speech.

## Notes
- Models are downloaded on first run (requires internet).
- The app runs on CPU by default; modify model files for GPU support if needed.
- Temporary files are saved in static/.
